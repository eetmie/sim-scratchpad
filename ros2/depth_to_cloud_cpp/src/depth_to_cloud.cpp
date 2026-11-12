// Depth image -> self-filtered point cloud, for FastDEM. Same node on the sim and the real robot.
//
// Per camera: subscribes /excavator/cameras/<name>/depth/{image_raw,camera_info} (16UC1 mm), samples every
// `stride`-th pixel, keeps 0 < |P| < max_range_m, and drops points inside any robot link's padded box. Boxes come
// as parameters (box_links, box_data: center xyz, axes as rows, half extents xyz per link); mapping.launch.py
// computes them from the URDF meshes (robot_boxes.py). Boxes are placed with TF at the image's own stamp, so a
// moving boom is cut where it was when the image was taken. Publishes .../points (kept, camera frame) and
// .../points_self (dropped, only when subscribed). Never publishes unfiltered points: no TF, no cloud.
//
// Optional stereo depth correction per camera, the model sand_height.py fits on flat sand (the "dome"):
// z' = z / (1 - z k(u, v)), k = sum coeffs . x^i y^j (i + j <= degree, in that order), x = (u - w/2) / (w/2),
// y = (v - h/2) / (w/2) over the physical sensor pixel (mirrored back when the image is published turned 180 deg).
// Parameters <camera>.correction_degree, <camera>.correction_coeffs (empty = off), <camera>.rotated_180.
//
// shortcut: one oriented box per link is loose for the bent boom (cuts air below the bend); upgrade to a distance
// to the mesh (like the robot's arm_model.npz) if terrain under the boom goes missing from the map.

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <stdexcept>
#include <string>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <vector>

using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

namespace {

struct Box {
  std::string link;
  Eigen::Vector3f center, half;  // half already includes the padding
  Eigen::Matrix3f axes;          // principal axes as rows: local = axes * (p_link - center)
};

struct Camera {
  int correction_degree = 0;
  std::vector<double> correction_coeffs;  // empty: no depth correction
  bool rotated_180 = false;
  std::vector<float> k;                   // per published pixel, built on the first image
  rclcpp::Publisher<PointCloud2>::SharedPtr kept, self;
  rclcpp::Subscription<CameraInfo>::SharedPtr info_sub;
  rclcpp::Subscription<Image>::SharedPtr depth_sub;
  CameraInfo::ConstSharedPtr info;
  int64_t last_ns = 0;
  bool seen = false;
};

std::string topic(const std::string& camera, const std::string& leaf) {
  return "/excavator/cameras/" + camera + "/" + leaf;
}

// k(u, v) per published pixel, exactly as sand_height.py's depth_basis(...) @ coeffs.
std::vector<float> correctionTable(const Camera& c, uint32_t w, uint32_t h) {
  const size_t terms = (c.correction_degree + 1) * (c.correction_degree + 2) / 2;
  if (c.correction_coeffs.size() != terms)
    throw std::runtime_error("correction_coeffs has " + std::to_string(c.correction_coeffs.size()) +
                             " terms; degree " + std::to_string(c.correction_degree) + " needs " +
                             std::to_string(terms));
  std::vector<float> k(static_cast<size_t>(w) * h);
  for (uint32_t v = 0; v < h; ++v)
    for (uint32_t u = 0; u < w; ++u) {
      const double su = c.rotated_180 ? w - 1.0 - u : u, sv = c.rotated_180 ? h - 1.0 - v : v;
      const double x = (su - w / 2.0) / (w / 2.0), y = (sv - h / 2.0) / (w / 2.0);
      double sum = 0.0;
      size_t n = 0;
      for (int i = 0; i <= c.correction_degree; ++i)
        for (int j = 0; j <= c.correction_degree - i; ++j) sum += c.correction_coeffs[n++] * std::pow(x, i) * std::pow(y, j);
      k[static_cast<size_t>(v) * w + u] = static_cast<float>(sum);
    }
  return k;
}

PointCloud2 cloud(const std_msgs::msg::Header& header, const std::vector<float>& xyz) {
  PointCloud2 msg;
  msg.header = header;
  msg.height = 1;
  msg.width = xyz.size() / 3;
  for (int i = 0; i < 3; ++i) {
    PointField f;
    f.name = std::string(1, "xyz"[i]);
    f.offset = 4 * i;
    f.datatype = PointField::FLOAT32;
    f.count = 1;
    msg.fields.push_back(f);
  }
  msg.is_bigendian = false;
  msg.point_step = 12;
  msg.row_step = 12 * msg.width;
  msg.is_dense = true;
  msg.data.resize(xyz.size() * sizeof(float));
  std::memcpy(msg.data.data(), xyz.data(), msg.data.size());
  return msg;
}

}  // namespace

class DepthToCloud : public rclcpp::Node {
 public:
  // Wall clock for the buffer: it only times the lookup wait, and with the sim clock a /clock jump back (Stop in
  // Isaac) clears the buffer, static camera frames included, which are never sent again.
  DepthToCloud()
      : Node("depth_to_cloud"),
        tf_buffer_(std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME)),
        tf_listener_(tf_buffer_) {
    stride_ = declare_parameter("stride", 2);
    max_range_ = declare_parameter("max_range_m", 0.9);
    const double padding = declare_parameter("padding_m", 0.025);
    // FastDEM integrates every cloud it gets; throttle here, by image stamp, so no backlog builds up.
    min_period_ns_ = static_cast<int64_t>(1e9 / declare_parameter("max_rate_hz", 5.0));
    const auto cameras = declare_parameter("cameras", std::vector<std::string>{"cabin"});
    const auto links = declare_parameter("box_links", std::vector<std::string>{});
    const auto data = declare_parameter("box_data", std::vector<double>{});
    if (links.empty() || data.size() != 15 * links.size())
      throw std::runtime_error("box_links / box_data missing or mismatched (mapping.launch.py sets them)");
    for (size_t i = 0; i < links.size(); ++i) {
      const double* d = &data[15 * i];
      Box b;
      b.link = links[i];
      b.center = Eigen::Vector3d(d[0], d[1], d[2]).cast<float>();
      b.axes = Eigen::Map<const Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(d + 3).cast<float>();
      b.half = (Eigen::Vector3d(d[12], d[13], d[14]).array() + padding).matrix().cast<float>();
      boxes_.push_back(b);
    }
    for (const auto& name : cameras) {
      Camera& c = cameras_[name];
      c.correction_degree = declare_parameter(name + ".correction_degree", 3);
      c.correction_coeffs = declare_parameter(name + ".correction_coeffs", std::vector<double>{});
      c.rotated_180 = declare_parameter(name + ".rotated_180", false);
      if (!c.correction_coeffs.empty()) {
        correctionTable(c, 2, 2);  // validates degree vs. coefficient count now, not on the first image
        RCLCPP_INFO(get_logger(), "%s: depth correction, degree %d", name.c_str(), c.correction_degree);
      }
      c.kept = create_publisher<PointCloud2>(topic(name, "points"), 2);
      c.self = create_publisher<PointCloud2>(topic(name, "points_self"), 2);
      c.info_sub = create_subscription<CameraInfo>(topic(name, "depth/camera_info"), 2,
                                                   [this, name](CameraInfo::ConstSharedPtr m) { cameras_[name].info = m; });
      c.depth_sub = create_subscription<Image>(topic(name, "depth/image_raw"), 1,
                                               [this, name](Image::ConstSharedPtr m) { onDepth(name, *m); });
    }
    RCLCPP_INFO(get_logger(), "%zu self-filter boxes, padding %.3f m, stride %ld", boxes_.size(), padding, stride_);
  }

 private:
  void onDepth(const std::string& name, const Image& msg) {
    Camera& c = cameras_.at(name);
    if (!c.info || msg.encoding != "16UC1") return;
    const int64_t stamp_ns = rclcpp::Time(msg.header.stamp).nanoseconds();
    // 0.9: tolerate jitter at exactly 5 Hz; llabs: sim time restarts at 0 after Stop.
    if (c.seen && std::llabs(stamp_ns - c.last_ns) < 0.9 * min_period_ns_) return;
    c.seen = true;
    c.last_ns = stamp_ns;
    const auto start = std::chrono::steady_clock::now();

    // Box frames in the camera frame at the image stamp: local = R * p_cam + t.
    std::vector<Eigen::Matrix3f> rot(boxes_.size());
    std::vector<Eigen::Vector3f> trans(boxes_.size());
    for (size_t i = 0; i < boxes_.size(); ++i) {
      try {
        const auto tf = tf_buffer_.lookupTransform(boxes_[i].link, msg.header.frame_id, msg.header.stamp,
                                                   rclcpp::Duration::from_seconds(0.3));
        const Eigen::Isometry3f link_cam = tf2::transformToEigen(tf).cast<float>();
        rot[i] = boxes_[i].axes * link_cam.linear();
        trans[i] = boxes_[i].axes * (link_cam.translation() - boxes_[i].center);
      } catch (const tf2::TransformException& e) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "%s: no TF %s <- %s at the image stamp: %s",
                             name.c_str(), boxes_[i].link.c_str(), msg.header.frame_id.c_str(), e.what());
        return;  // never publish unfiltered points: the boom would become terrain
      }
    }
    const auto tf_done = std::chrono::steady_clock::now();

    if (!c.correction_coeffs.empty() && c.k.size() != static_cast<size_t>(msg.width) * msg.height)
      c.k = correctionTable(c, msg.width, msg.height);
    const float* corr = c.k.empty() ? nullptr : c.k.data();
    const auto& K = c.info->k;
    const float fx = K[0], cx = K[2], fy = K[4], cy = K[5], max_sq = max_range_ * max_range_;
    std::vector<float> kept, self;
    kept.reserve(3 * (msg.width / stride_ + 1) * (msg.height / stride_ + 1));
    for (uint32_t v = 0; v < msg.height; v += stride_) {
      const auto* row = reinterpret_cast<const uint16_t*>(&msg.data[v * msg.step]);
      for (uint32_t u = 0; u < msg.width; u += stride_) {
        if (!row[u]) continue;
        float z = row[u] * 0.001f;
        if (corr) {
          const float denom = 1.0f - z * corr[v * msg.width + u];
          if (denom <= 0.0f) continue;  // outside the fitted model
          z /= denom;
        }
        const Eigen::Vector3f p((u - cx) / fx * z, (v - cy) / fy * z, z);
        if (p.squaredNorm() >= max_sq) continue;
        bool robot = false;
        for (size_t i = 0; i < boxes_.size() && !robot; ++i)
          robot = ((rot[i] * p + trans[i]).cwiseAbs().array() <= boxes_[i].half.array()).all();
        auto& out = robot ? self : kept;
        out.insert(out.end(), {p.x(), p.y(), p.z()});
      }
    }
    c.kept->publish(cloud(msg.header, kept));
    if (c.self->get_subscription_count()) c.self->publish(cloud(msg.header, self));

    using ms = std::chrono::duration<double, std::milli>;
    total_ms_ += ms(std::chrono::steady_clock::now() - start).count();
    tf_ms_ += ms(tf_done - start).count();
    ++count_;
    const double since = std::chrono::duration<double>(std::chrono::steady_clock::now() - log_since_).count();
    if (since > 10.0) {
      RCLCPP_INFO(get_logger(), "%.1f images/s, %.1f ms each (%.1f ms waiting for TF)", count_ / since,
                  total_ms_ / count_, tf_ms_ / count_);
      total_ms_ = tf_ms_ = 0.0;
      count_ = 0;
      log_since_ = std::chrono::steady_clock::now();
    }
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;  // own node + thread, so lookups can wait inside this callback
  std::vector<Box> boxes_;
  std::map<std::string, Camera> cameras_;
  int64_t stride_;
  double max_range_;
  int64_t min_period_ns_;
  double total_ms_ = 0.0, tf_ms_ = 0.0;
  int count_ = 0;
  std::chrono::steady_clock::time_point log_since_ = std::chrono::steady_clock::now();
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DepthToCloud>());
  rclcpp::shutdown();
  return 0;
}
