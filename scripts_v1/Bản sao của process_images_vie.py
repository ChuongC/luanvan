import os
import json
from PIL import Image
from collections import defaultdict
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from dataset_collection.scrape_utils import apply_instructions

def process_images_from_instructions(instructions_file, source_folder, target_folder, output_json):
    os.makedirs(target_folder, exist_ok=True)
    processed_info = []

    processed_count = 0
    skipped_count = 0
    error_count = 0
    instruction_type_counter = defaultdict(int)

    # Đọc hướng dẫn từ file nếu có
    instructions_map = {}
    if os.path.exists(instructions_file):
        with open(instructions_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line or ": " not in line:
                    continue
                image_name, instructions = line.split(": ", 1)
                instructions_map[image_name] = instructions

    all_images = sorted(os.listdir(source_folder))
    total_images = len(all_images)

    for image_name in all_images:
        image_path = os.path.join(source_folder, image_name)

        if not os.path.isfile(image_path):
            continue

        # Lấy hướng dẫn nếu có, ngược lại dùng mặc định
        instructions = instructions_map.get(image_name, "Standard processing")

        # Thống kê loại chỉ dẫn
        instruction_parts = [instr.strip() for instr in instructions.split(";")]
        for part in instruction_parts:
            if part:
                instr_type = part.split()[0].lower()
                instruction_type_counter[instr_type] += 1

        try:
            image = Image.open(image_path)
            processed_image = apply_instructions(image, instructions)
            target_image_path = os.path.join(target_folder, image_name)
            processed_image.save(target_image_path)

            # So sánh ảnh gốc và ảnh xử lý
            original_image = Image.open(image_path)
            size_str = f"{original_image.size[0]}x{original_image.size[1]} → {processed_image.size[0]}x{processed_image.size[1]}"
            mode_str = f"{original_image.mode} → {processed_image.mode}" if original_image.mode != processed_image.mode else original_image.mode
            format_str = f"{(original_image.format or 'N/A')} → {(processed_image.format or 'N/A')}" if original_image.format != processed_image.format else (original_image.format or 'N/A')
            byte_str = f"{os.path.getsize(image_path)}B → {os.path.getsize(target_image_path)}B"

            print(f"📌 {image_name}: size {size_str}, mode {mode_str}, format {format_str}, file {byte_str}")
            print(f"✅ Đã xử lý và lưu: {image_name}")

            processed_info.append({
                "image": image_name,
                "instruction": instructions,
                "output_path": target_image_path
            })
            processed_count += 1

        except Exception as e:
            print(f"❌ Lỗi xử lý {image_name}: {e}")
            error_count += 1

    # Ghi file JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(processed_info, f, ensure_ascii=False, indent=2)

    print(f"\n📝 Đã ghi thông tin xử lý vào {output_json}")
    print(f"\n📊 THỐNG KÊ:")
    print(f"- Tổng số ảnh trong thư mục           : {total_images}")
    print(f"- ✅ Số ảnh xử lý thành công            : {processed_count}")
    print(f"- ⚠️ Số ảnh bị bỏ qua                   : {skipped_count}")
    print(f"- ❌ Số ảnh lỗi trong quá trình xử lý   : {error_count}")
    print("\n📌 Thống kê loại chỉ dẫn đã dùng:")
    for instr_type, count in sorted(instruction_type_counter.items()):
        print(f"  - {instr_type}: {count}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Xử lý ảnh theo hướng dẫn, mặc định xử lý toàn bộ ảnh trong thư mục")
    parser.add_argument('--instruction_file', type=str, default='dataset/image_processing_instructions.txt')
    parser.add_argument('--source_folder', type=str, default='dataset/img/')
    parser.add_argument('--target_folder', type=str, default='dataset/processed_img_vie/')
    parser.add_argument('--output_json', type=str, default='dataset/processed_img_vie.json')

    args = parser.parse_args()

    process_images_from_instructions(
        instructions_file=args.instruction_file,
        source_folder=args.source_folder,
        target_folder=args.target_folder,
        output_json=args.output_json
    )
