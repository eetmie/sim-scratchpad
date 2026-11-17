# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small OpenGL textures for sensor previews; upload only when a new scan arrives."""

import ctypes

import numpy as np


class ImageTextures:
    def __init__(self):
        self.images = {}

    def draw(self, imgui, name, gray, generation, size):
        from pyglet import gl

        if name not in self.images:
            texture = gl.GLuint()
            gl.glGenTextures(1, ctypes.byref(texture))
            self.images[name] = [texture, -1]
        texture, previous = self.images[name]
        if previous != generation:
            rgba = np.empty((*gray.shape[:2], 4), dtype=np.uint8)
            rgba[..., :3] = gray[..., None] if gray.ndim == 2 else gray  # mono8 or rgb8
            rgba[..., 3] = 255
            old_texture = gl.GLint()
            gl.glGetIntegerv(gl.GL_TEXTURE_BINDING_2D, ctypes.byref(old_texture))
            gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_RGBA8,
                gray.shape[1],
                gray.shape[0],
                0,
                gl.GL_RGBA,
                gl.GL_UNSIGNED_BYTE,
                rgba.ctypes.data_as(ctypes.c_void_p),
            )
            gl.glBindTexture(gl.GL_TEXTURE_2D, old_texture.value)
            self.images[name][1] = generation
        imgui.image(imgui.ImTextureRef(texture.value), imgui.ImVec2(*size))

    def close(self):
        if self.images:
            from pyglet import gl

            for texture, _ in self.images.values():
                gl.glDeleteTextures(1, ctypes.byref(texture))
        self.images.clear()
