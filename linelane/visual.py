import cv2
import os
import argparse
import glob
from tqdm import tqdm

def visualize_dataset(input_dir, output_dir):
    """
    데이터셋의 원본 이미지에 컬러맵을 오버레이하여 저장하는 함수.

    Args:
        input_dir (str): 'image', 'colormap' 폴더가 있는 원본 데이터셋 경로.
        output_dir (str): 시각화 결과물을 저장할 경로.
    """
    # 1. 입력 및 출력 경로 설정 및 확인
    image_dir = os.path.join(input_dir, 'image')
    colormap_dir = os.path.join(input_dir, 'colormap')

    if not os.path.isdir(image_dir) or not os.path.isdir(colormap_dir):
        print(f"오류: 입력 폴더 '{input_dir}' 안에 'image' 또는 'colormap' 폴더가 없습니다.")
        print("-> 데이터셋 폴더 경로를 정확히 입력해주세요.")
        return

    # 출력 폴더 생성
    os.makedirs(output_dir, exist_ok=True)
    print(f"결과물은 '{output_dir}' 폴더에 저장됩니다.")

    # 2. 처리할 이미지 파일 목록 가져오기
    # glob을 사용하여 png와 jpg 파일을 모두 찾습니다.
    image_files = glob.glob(os.path.join(image_dir, '*.png'))
    image_files.extend(glob.glob(os.path.join(image_dir, '*.jpg')))
    
    if not image_files:
        print(f"'{image_dir}' 폴더에 처리할 이미지가 없습니다.")
        return
        
    # 파일 이름을 기준으로 정렬하여 순서를 보장합니다.
    image_files.sort()

    # 3. 각 이미지에 대해 오버레이 작업 수행
    # tqdm을 사용하여 진행 상황을 시각적으로 표시합니다.
    for image_path in tqdm(image_files, desc="시각화 진행 중"):
        try:
            # 파일 이름 추출 (예: 000123.png)
            base_filename = os.path.basename(image_path)
            
            # 해당 컬러맵 파일 경로 생성
            colormap_path = os.path.join(colormap_dir, base_filename)

            # 원본 이미지와 컬러맵 이미지 불러오기
            original_image = cv2.imread(image_path)
            colormap_image = cv2.imread(colormap_path)

            # 파일이 존재하지 않거나 읽기 실패 시 건너뛰기
            if original_image is None or colormap_image is None:
                print(f"경고: {base_filename} 또는 해당하는 컬러맵 파일을 읽을 수 없어 건너뜁니다.")
                continue

            # 이미지 크기가 다를 경우 리사이징 (보통은 크기가 같음)
            if original_image.shape != colormap_image.shape:
                colormap_image = cv2.resize(colormap_image, (original_image.shape[1], original_image.shape[0]))

            # 4. 이미지 합성 (Blending)
            # alpha: 원본 이미지 가중치, beta: 컬러맵 이미지 가중치
            alpha = 0.6
            beta = 0.4
            gamma = 0.0 # 밝기 조절값
            
            blended_image = cv2.addWeighted(original_image, alpha, colormap_image, beta, gamma)

            # 5. 결과물 저장
            output_path = os.path.join(output_dir, base_filename)
            # cv2.imwrite(output_path, blended_image)
            cv2.imshow("image", blended_image)
            cv2.waitKey()

        except Exception as e:
            print(f"파일 처리 중 오류 발생: {image_path}, 오류: {e}")

    print("\n✅ 모든 이미지의 시각화 및 저장이 완료되었습니다.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CARLA 데이터셋의 원본 이미지에 컬러맵을 오버레이합니다.")
    parser.add_argument(
        '--input', 
        type=str, 
        default='drivable_multiclass_dataset',
        help="이미지와 컬러맵 폴더가 포함된 원본 데이터셋 폴더 경로"
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='visualization_output',
        help="오버레이된 결과 이미지를 저장할 폴더 경로"
    )
    args = parser.parse_args()

    visualize_dataset(args.input, args.output)