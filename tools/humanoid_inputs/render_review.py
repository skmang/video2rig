"""通过 Blender 公开渲染钩子提高 JPEG 质量；渲染、坐标和标签仍由原版生成。"""
from pathlib import Path
import runpy
import bpy


def high_quality_jpeg(scene, *args):
    scene.render.image_settings.quality = 100


bpy.app.handlers.render_pre.append(high_quality_jpeg)
original = Path(__file__).resolve().parents[2] / 'blender' / 'render_poses.py'
runpy.run_path(str(original), run_name='__main__')
