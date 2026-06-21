 #!/usr/bin/env python3
 """
 ═══════════════════════════════════════════════════════════════════════
  咕嘎 & doro · 雪地跳舞视频生成流水线
  ──────────────────────────────────────────────────────────────────────
  Step 1 — 文生图 (DALL-E 3)     → 生成原始场景
  Step 2 — 图编图 (gpt-image-2)  → 二次修改优化
  Step 3 — 图生视频 (OpenAI Sora) → 生成舞蹈视频 + 状态轮询
 ═══════════════════════════════════════════════════════════════════════
 
 使用方法:
   1. 安装依赖:  pip install openai requests pillow
   2. 设置环境变量:  set OPENAI_API_KEY=sk-...
   3. 运行脚本:  python pipeline.py
   4. 或者指定输出目录:  python pipeline.py --output-dir ./output
 
 要求:
   - Python 3.9+
   - OpenAI API Key （需要 DALL-E 3 + Sora 访问权限）
 ═══════════════════════════════════════════════════════════════════════
 """
 
 from __future__ import annotations
 
 import argparse
 import base64
 import json
 import os
 import sys
 import time
 import logging
 from pathlib import Path
 from typing import Any, Optional
 from io import BytesIO
 
 # ── 日志配置 ──────────────────────────────────────────────────────────
 logging.basicConfig(
     level=logging.INFO,
     format="%(asctime)s | %(levelname)-7s | %(message)s",
     datefmt="%H:%M:%S",
 )
 log = logging.getLogger("pipeline")
 
 # ── 常量 ──────────────────────────────────────────────────────────────
 DEFAULT_OUTPUT_DIR = Path("./output/snow-dance")
 SORA_POLL_INTERVAL = 10       # 秒
 SORA_MAX_POLL_TIME = 600      # 最长等待 10 分钟
 MAX_RETRIES = 3
 
 # =====================================================================
 #  Step 1: 文生图 — DALL-E 3
 # =====================================================================
 def step1_generate_image(client: Any, output_dir: Path) -> Path:
     """调用 DALL-E 3 生成原始场景图"""
     log.info("=" * 55)
     log.info("Step 1/3: 文生图 (DALL-E 3)")
     log.info("=" * 55)
 
     prompt = (
         "两个可爱的卡通角色在雪夜中手牵手跳舞。"
         "左边是一个绿色的小角色叫「咕嘎」，圆圆的头戴红色围巾；"
         "右边是一个橙色的小角色叫「doro」，戴蓝色围巾。"
         "它们站在雪地上，空中飘着雪花，背景有月亮和星光。"
         "温馨浪漫的冬季场景。插画风格，色彩温暖明亮。"
     )
 
     payload = {
         "model": "dall-e-3",
         "prompt": prompt,
         "n": 1,
         "size": "1792x1024",
         "quality": "standard",
         "response_format": "b64_json",
     }
 
     log.info("正在生成原始图片…（DALL-E 3 通常需要 30-60s）")
     start = time.time()
 
     for attempt in range(1, MAX_RETRIES + 1):
         try:
             resp = client.images.generate(**payload)
             break
         except Exception as exc:
             if attempt == MAX_RETRIES:
                 log.error("DALL-E 3 调用失败，已达最大重试次数: %s", exc)
                 raise
             wait = min(2 ** attempt * 5, 60)
             log.warning("第 %d 次重试 (%s)，%ds 后重试…", attempt, exc, wait)
             time.sleep(wait)
 
     elapsed = time.time() - start
     log.info("DALL-E 3 生成完成，耗时 %.1fs", elapsed)
 
     image_b64 = resp.data[0].b64_json
     out_path = output_dir / "01-original.png"
     out_path.parent.mkdir(parents=True, exist_ok=True)
     out_path.write_bytes(base64.b64decode(image_b64))
     log.info("原始图片已保存: %s", out_path)
 
     return out_path
 
 
 # =====================================================================
 #  Step 2: 图编图 — gpt-image-2
 # =====================================================================
 def step2_edit_image(client: Any, input_path: Path, output_dir: Path) -> Path:
     """对原始图进行二次修改：优化角色细节、氛围等"""
     log.info("=" * 55)
     log.info("Step 2/3: 图编图 (gpt-image-2)")
     log.info("=" * 55)
 
     prompt = (
         "优化这张插画：让两个角色面部表情更可爱、笑容更明显；"
         "雪花效果更丰富；地面雪地更蓬松有光泽；"
         "加入一点暖色光晕让氛围更温馨。"
         "保持原有的角色设计和构图不变。"
     )
 
     log.info("正在编辑优化图片…（gpt-image-2 通常需要 15-30s）")
     start = time.time()
 
     for attempt in range(1, MAX_RETRIES + 1):
         try:
             with open(input_path, "rb") as f:
                 image_data = f.read()
 
             resp = client.images.edit(
                 model="gpt-image-2",
                 prompt=prompt,
                 image=image_data,
                 n=1,
                 size="auto",
                 quality="high",
                 response_format="b64_json",
             )
             break
         except Exception as exc:
             if attempt == MAX_RETRIES:
                 log.error("图片编辑失败，已达最大重试次数: %s", exc)
                 raise
             wait = min(2 ** attempt * 5, 60)
             log.warning("第 %d 次重试 (%s)，%ds 后重试…", attempt, exc, wait)
             time.sleep(wait)
 
     elapsed = time.time() - start
     log.info("图片编辑完成，耗时 %.1fs", elapsed)
 
     image_b64 = resp.data[0].b64_json
     out_path = output_dir / "02-edited.png"
     out_path.write_bytes(base64.b64decode(image_b64))
     log.info("编辑后图片已保存: %s", out_path)
 
     return out_path
 
 
 # =====================================================================
 #  Step 3: 图生视频 — OpenAI Sora (异步 + 状态轮询)
 # =====================================================================
 def step3_image_to_video(client: Any, input_path: Path, output_dir: Path) -> Path:
     """将编辑后的图片作为输入，调用 Sora API 生成视频，轮询直到完成"""
     log.info("=" * 55)
     log.info("Step 3/3: 图生视频 (Sora)")
     log.info("=" * 55)
 
     # ── 3a: 将图片读为 base64 ──
     with open(input_path, "rb") as f:
         image_b64 = base64.b64encode(f.read()).decode("utf-8")
 
     prompt = (
         "Two cute cartoon characters, a green one named Guga and an orange one named Doro, "
         "holding hands and dancing joyfully in a snowy night scene. "
         "Snowflakes falling gently, moonlight shining. "
         "Warm and romantic atmosphere. Smooth looping dance motion."
     )
 
     # ── 3b: 创建视频生成任务 ──
     log.info("正在创建 Sora 视频生成任务…")
     log.info("提示词: %s", prompt)
 
     generation_id = _create_video_generation(
         client=client,
         prompt=prompt,
         image_b64=image_b64,
     )
     log.info("视频生成任务已创建，ID: %s", generation_id)
 
     # ── 3c: 轮询直到完成 ──
     log.info("轮询等待视频生成完成（每 %ds 检查一次，最长等待 %ds）…",
               SORA_POLL_INTERVAL, SORA_MAX_POLL_TIME)
     start = time.time()
 
     video_url = _poll_video_generation(client, generation_id)
 
     elapsed = time.time() - start
     log.info("视频生成完成，耗时 %.1fs", elapsed)
 
     # ── 3d: 下载视频 ──
     out_path = output_dir / "03-final-video.mp4"
     out_path.parent.mkdir(parents=True, exist_ok=True)
     _download_video(video_url, out_path)
     log.info("最终视频已保存: %s", out_path)
     log.info("文件大小: %.1f MB", out_path.stat().st_size / (1024 * 1024))
 
     return out_path
 
 
 # ── Sora API 辅助函数 ────────────────────────────────────────────────
 
 def _create_video_generation(
     client: Any,
     prompt: str,
     image_b64: str,
 ) -> str:
     """
     调用 Sora API 创建图生视频任务。
     
     注意：Sora API 的实际端点可能在 OpenAI SDK 演进中变化。
     如果 client.video 不可用，脚本会降级为直接 HTTP 调用。
     """
     # 方式 A: 通过 OpenAI SDK（推荐，如果库版本支持）
     try:
         if hasattr(client, "video") and hasattr(client.video, "generations"):
             resp = client.video.generations.create(
                 model="sora-v1",
                 prompt=prompt,
                 image=image_b64,
             )
             return resp.id
     except AttributeError:
         log.info("SDK 未暴露 video 端点，降级为直接 HTTP 调用")
     except Exception as exc:
         log.warning("SDK video 调用失败，降级为 HTTP: %s", exc)
 
     # 方式 B: 直接 HTTP 调用
     import httpx  # 延迟导入
 
     api_key = os.environ["OPENAI_API_KEY"]
     headers = {
         "Authorization": f"Bearer {api_key}",
         "Content-Type": "application/json",
     }
     payload = {
         "model": "sora-v1",
         "prompt": prompt,
         "image": image_b64,
         "n": 1,
     }
 
     for attempt in range(1, MAX_RETRIES + 1):
         try:
             resp = httpx.post(
                 "https://api.openai.com/v1/video/generations",
                 headers=headers,
                 json=payload,
                 timeout=120,
             )
             resp.raise_for_status()
             data = resp.json()
             gen_id = data.get("id")
             if not gen_id:
                 raise RuntimeError(
                     f"Sora 响应中没有 generation ID: {json.dumps(data, indent=2)[:500]}"
                 )
             return gen_id
         except httpx.HTTPStatusError as exc:
             if attempt == MAX_RETRIES:
                 raise RuntimeError(
                     f"Sora API 错误 (status {exc.response.status_code}): "
                     f"{exc.response.text[:500]}"
                 ) from exc
             wait = min(2 ** attempt * 5, 60)
             log.warning("创建视频任务第 %d 次重试，%ds 后重试…", attempt, wait)
             time.sleep(wait)
         except (httpx.RequestError, httpx.TimeoutException) as exc:
             if attempt == MAX_RETRIES:
                 raise RuntimeError(f"Sora 网络请求失败: {exc}") from exc
             wait = min(2 ** attempt * 5, 60)
             log.warning("网络错误第 %d 次重试，%ds 后重试…", attempt, wait)
             time.sleep(wait)
 
     raise RuntimeError("无法创建 Sora 视频生成任务")
 
 
 def _poll_video_generation(client: Any, generation_id: str) -> str:
     """
     轮询 Sora API，直到视频生成完成。
     返回下载 URL。
     """
     import httpx
 
     api_key = os.environ["OPENAI_API_KEY"]
     headers = {"Authorization": f"Bearer {api_key}"}
     start_time = time.time()
     last_log = 0
 
     while True:
         elapsed = time.time() - start_time
         if elapsed > SORA_MAX_POLL_TIME:
             raise TimeoutError(
                 f"Sora 视频生成超时（超过 {SORA_MAX_POLL_TIME}s）"
             )
 
         try:
             resp = httpx.get(
                 f"https://api.openai.com/v1/video/generations/{generation_id}",
                 headers=headers,
                 timeout=30,
             )
             resp.raise_for_status()
             data = resp.json()
         except Exception as exc:
             log.warning("轮询请求失败: %s，%ds 后重试…", exc, SORA_POLL_INTERVAL)
             time.sleep(SORA_POLL_INTERVAL)
             continue
 
         status = data.get("status", "unknown")
 
         # 每 30 秒打印一次进度
         if elapsed - last_log > 30 or status in ("completed", "failed"):
             log.info("视频生成状态: %s （已等待 %.0fs）", status, elapsed)
             last_log = elapsed
 
         if status == "completed":
             # 提取视频 URL —— 具体字段名视 API 版本而定
             video = data.get("video") or data.get("output") or data
             if isinstance(video, dict):
                 video_url = video.get("url") or video.get("video_url") or data.get("url")
             elif isinstance(video, str):
                 video_url = video
             else:
                 video_url = data.get("url")
 
             if not video_url:
                 raise RuntimeError(
                     "Sora 返回 completed 但未找到视频 URL。"
                     f"完整响应: {json.dumps(data, indent=2)[:1000]}"
                 )
             return video_url
 
         elif status == "failed":
             error = data.get("error", {}).get("message", json.dumps(data, ensure_ascii=False)[:500])
             raise RuntimeError(f"Sora 视频生成失败: {error}")
 
         elif status in ("in_progress", "queued", "processing"):
             time.sleep(SORA_POLL_INTERVAL)
 
         else:
             log.warning("未知状态 '%s'，继续轮询…", status)
             time.sleep(SORA_POLL_INTERVAL)
 
 
 def _download_video(url: str, out_path: Path) -> None:
     """下载视频文件到本地"""
     import httpx
 
     log.info("正在下载视频… (这可能是一个大文件)")
     start = time.time()
 
     with httpx.Client(timeout=300, follow_redirects=True) as hx:
         resp = hx.get(url)
         resp.raise_for_status()
         out_path.write_bytes(resp.content)
 
     elapsed = time.time() - start
     log.info("视频下载完成，耗时 %.1fs", elapsed)
 
 
 # =====================================================================
 #  CLI 入口
 # =====================================================================
 def create_openai_client() -> Any:
     """创建 OpenAI 客户端，含清晰的 Key 检测"""
     api_key = os.environ.get("OPENAI_API_KEY")
     if not api_key:
         log.error("未设置 OPENAI_API_KEY 环境变量")
         log.error("请先执行:  set OPENAI_API_KEY=sk-...  (Windows)")
         log.error("        export OPENAI_API_KEY=sk-...  (macOS/Linux)")
         sys.exit(1)
 
     try:
         from openai import OpenAI
     except ImportError:
         log.error("缺少 openai 库。请运行:  pip install openai")
         sys.exit(1)
 
     return OpenAI(api_key=api_key)
 
 
 def parse_args() -> argparse.Namespace:
     parser = argparse.ArgumentParser(
         description="咕嘎 & doro 雪地跳舞视频生成流水线",
         formatter_class=argparse.RawDescriptionHelpFormatter,
         epilog="""
示例:
  python pipeline.py
  python pipeline.py --output-dir ./my-video --skip-edit
  python pipeline.py --resume-from 02-edited.png
        """,
     )
     parser.add_argument(
         "--output-dir",
         type=Path,
         default=DEFAULT_OUTPUT_DIR,
         help=f"输出目录（默认: {DEFAULT_OUTPUT_DIR}）",
     )
     parser.add_argument(
         "--skip-edit",
         action="store_true",
         help="跳过 Step 2 编辑，直接用 DALL-E 3 的原始图生成视频",
     )
     parser.add_argument(
         "--resume-from",
         type=Path,
         help="跳转到指定步骤的图片开始（如 --resume-from 02-edited.png）",
     )
     return parser.parse_args()
 
 
 def main() -> None:
     args = parse_args()
     output_dir = args.output_dir
     output_dir.mkdir(parents=True, exist_ok=True)
 
     client = create_openai_client()
 
     # ── Step 1: 文生图 ──
     if args.resume_from:
         log.info("跳过 Step 1，从已有图片开始: %s", args.resume_from)
         if not args.resume_from.exists():
             log.error("指定的图片不存在: %s", args.resume_from)
             sys.exit(1)
         raw_path = args.resume_from
     else:
         raw_path = step1_generate_image(client, output_dir)
 
     # ── Step 2: 图编图（可选） ──
     if args.skip_edit or args.resume_from:
         if args.skip_edit:
             log.info("跳过 Step 2（--skip-edit）")
         edited_path = raw_path
     else:
         edited_path = step2_edit_image(client, raw_path, output_dir)
 
     # ── Step 3: 图生视频 ──
     try:
         video_path = step3_image_to_video(client, edited_path, output_dir)
     except (TimeoutError, RuntimeError) as exc:
         log.error("Step 3 失败: %s", exc)
         log.info("可以稍后使用 --resume-from %s 重试 Step 3", edited_path)
         sys.exit(1)
 
     # ── 完成 ──
     log.info("=" * 55)
     log.info("🎉 全部完成！视频文件: %s", video_path)
     log.info("📁 中间产物: %s", output_dir)
     log.info("=" * 55)
 
 
 if __name__ == "__main__":
     main()
