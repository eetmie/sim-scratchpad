#include <chrono>
#include <future>
#include <memory>
#include <optional>

#include <QElapsedTimer>
#include <QLabel>
#include <QPushButton>
#include <QTimer>
#include <QVBoxLayout>

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/panel.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <std_srvs/srv/trigger.hpp>

namespace excavator_rviz {

class TerrainPanel : public rviz_common::Panel {
  using Trigger = std_srvs::srv::Trigger;
  using Client = rclcpp::Client<Trigger>;

  void onInitialize() override {
    auto node = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();
    client_ = node->create_client<Trigger>("/fastdem/reset_map");
    auto *button = new QPushButton("Clear terrain", this);
    button->setToolTip("Clear remembered terrain and redraw from the current camera views");
    auto *status = new QLabel(this);
    status->setWordWrap(true);
    auto *layout = new QVBoxLayout(this);
    layout->addWidget(button);
    layout->addWidget(status);
    layout->addStretch();
    connect(button, &QPushButton::clicked, this, [this, button, status] {
      if (!client_->service_is_ready()) {
        status->setText("Waiting for the terrain mapper. Try again when it is running.");
        return;
      }
      pending_.emplace(client_->async_send_request(std::make_shared<Trigger::Request>()));
      button->setEnabled(false);
      status->setText("Requesting reset…");
      elapsed_.start();
      timer_.start(100);
    });
    connect(&timer_, &QTimer::timeout, this, [this, button, status] {
      if (pending_->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        auto reply = pending_->get();
        status->setText(QString::fromStdString(reply->message));
      } else if (elapsed_.elapsed() >= 3000) {
        client_->remove_pending_request(*pending_);
        status->setText("No reply from the terrain mapper.");
      } else {
        return;
      }
      timer_.stop();
      pending_.reset();
      button->setEnabled(true);
    });
  }

  Client::SharedPtr client_;
  std::optional<Client::FutureAndRequestId> pending_;
  QTimer timer_;
  QElapsedTimer elapsed_;
};

}  // namespace excavator_rviz

PLUGINLIB_EXPORT_CLASS(excavator_rviz::TerrainPanel, rviz_common::Panel)
