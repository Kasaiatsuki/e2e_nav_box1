import PIL.Image
from torch.utils.tensorboard import SummaryWriter

def apply_pillow_patch():
    """
    TensorBoardの画像表示におけるPillow 10+の互換性問題を解決するためのパッチ
    Pillow 10以降で 'ANTIALIAS' が削除されたため、属性が見つからない場合は 'LANCZOS' で代用する。
    """
    if not hasattr(PIL.Image, 'ANTIALIAS'):
        PIL.Image.ANTIALIAS = getattr(PIL.Image, 'LANCZOS', PIL.Image.BICUBIC)

class TensorBoardLogger:
    """TensorBoardへのログ出力を一元管理するクラス"""
    def __init__(self, log_dir):
        # 初期化時にパッチを適用
        apply_pillow_patch()
        # flush_secs=10 でリアルタイム性を向上
        self.writer = SummaryWriter(log_dir=str(log_dir), flush_secs=10)

    def add_images(self, tag, images, global_step):
        self.writer.add_images(tag, images, global_step)

    def add_scalar(self, tag, value, global_step):
        self.writer.add_scalar(tag, value, global_step)

    def close(self):
        self.writer.close()
