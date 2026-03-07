#!/usr/bin/env python3
import argparse
import cv2
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="指定した画像の明るさを変更して保存するスクリプト")
    parser.add_argument("image_path", type=str, help="処理する画像のディレクトリまたはファイルのパス")
    parser.add_argument("--beta", type=int, default=30, help="明るさの変動値（負の値で暗く、正の値で明るくなります。推奨: -50 ~ 50）")
    parser.add_argument("--output", type=str, default=None, help="保存先のファイルパス（指定しない場合は元のファイル名に '_beta{値}' が付与されます）")

    args = parser.parse_args()
    
    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"エラー: 指定されたファイル・ディレクトリが見つかりません -> {img_path}")
        sys.exit(1)

    # 単一ファイルの場合の処理関数
    def process_image(filepath: Path, output_path: Path = None):
        img = cv2.imread(str(filepath))
        if img is None:
            print(f"スキップ: 画像として読み込めません -> {filepath}")
            return
            
        # 明るさの変更 (betaを加算、alpha=1.0でコントラストはそのまま)
        adjusted_img = cv2.convertScaleAbs(img, alpha=1.0, beta=args.beta)

        if output_path is None:
            # e2e_nav_box1/image ディレクトリに保存
            image_dir = Path(__file__).resolve().parent / "image"
            output_path = image_dir / f"{filepath.stem}_beta{args.beta}{filepath.suffix}"
        elif output_path.is_dir() or str(output_path).endswith('/') or not output_path.suffix:
            # outputがディレクトリとして指定された場合、ファイル名を付与
            output_path = output_path / f"{filepath.stem}_beta{args.beta}{filepath.suffix}"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), adjusted_img)
        print(f"保存しました: {output_path}")

    if img_path.is_file():
        # 指定されたのがファイルのとき
        process_image(img_path, Path(args.output) if args.output else None)
    elif img_path.is_dir():
        # 指定されたのがディレクトリのとき（ディレクトリ内の全画像を処理）
        print(f"ディレクトリ内の全画像を処理します: {img_path}")
        count = 0
        for ext in ['.png', '.jpg', '.jpeg']:
            for file in img_path.glob(f'*{ext}'):
                process_image(file)
                count += 1
        print(f"合計 {count} 枚の画像を処理しました。")

if __name__ == "__main__":
    main()
