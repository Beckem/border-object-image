import os
import numpy as np
import cv2

HOOK_IMAGE_PATH = "circle.png"


# Configuration constants for border and hook parameters
CONFIG = {
  "hook_size_min": 40,  # Minimum size of the hook in pixels
  "hook_size_ratio": 0.08,  # Hook size as a percentage of image size
  "min_limit_x" : 0.4,  # Minimum limit for x coordinate of hook (% of image width)
  "max_limit_x" : 0.6,  # Maximum limit for x coordinate of hook (% of image width)
  "inner_radius_ratio": 0.25,  # Inner radius of hook border as a percentage of hook size
  "border_color": [64, 21, 243], # Border color in BGR format
}

def hex_to_bgr(hex_color: str) -> list:
    """
    Convert hex color string (e.g. '#f3202c') to BGR list for OpenCV.
    """
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        raise ValueError("Hex color must be 6 digits")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return [b, g, r]

def upscale_image(cv_image, scale_factor=2):
    """
    Upscale ảnh sử dụng OpenCV 
    Args:
        cv_image: numpy array (BGR hoặc BGRA)
        scale_factor: tỷ lệ phóng to (2, 3, 4)
    """
    print(f"Đang upscale ảnh {scale_factor}x")

    has_alpha = cv_image.shape[2] == 4 if len(cv_image.shape) == 3 else False
    bgr = cv_image[..., :3]

    h, w = bgr.shape[:2]
    upscaled_bgr = cv2.resize(bgr, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_CUBIC)

    if has_alpha:
        alpha = cv_image[..., 3]
        alpha_up = cv2.resize(alpha, (upscaled_bgr.shape[1], upscaled_bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        result = np.dstack([upscaled_bgr, alpha_up])
    else:
        result = upscaled_bgr

    return result

def processed_image(image_path: str, output_path: str, border_thickness_px: int, border_padding_px: int, border_color: str, show_hook: bool):
    reference_size = 1000
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        return {"status": "error", "message": "Không thể đọc ảnh"}
    
    if len(image.shape) != 3 or image.shape[2] != 4:
        return {"status": "error", "message": "Ảnh phải có 4 channels (BGRA)"}

    hook_image = cv2.imread(HOOK_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    
    if not os.path.exists(HOOK_IMAGE_PATH):
        return {"status": "error", "message": f"Không tìm thấy ảnh móc treo: {HOOK_IMAGE_PATH}"}
    
    if len(hook_image.shape) == 2:
        hook_image = cv2.cvtColor(hook_image, cv2.COLOR_GRAY2BGR)
    
    if len(hook_image.shape) == 3 and hook_image.shape[2] == 3:
        hook_alpha = np.ones((hook_image.shape[0], hook_image.shape[1], 1), dtype=np.uint8) * 255
        hook_image = np.concatenate([hook_image, hook_alpha], axis=2)
    elif len(hook_image.shape) == 3 and hook_image.shape[2] == 4:
        pass
    else:
        return {"status": "error", "message": f"Định dạng ảnh móc không được hỗ trợ: {hook_image.shape}"}
    
    h, w = image.shape[:2]
    print(f"Width x Height: {w}x{h}")
    area = w * h
    
    # Upscale và độ dày stroke dựa trên diện tích ảnh
    if area < 200000:
        scale_factor = 8
    elif area < 500000:
        scale_factor = 4
    elif area < 1000000:
        scale_factor = 3
    elif area < 2000000:
        scale_factor = 2
    else:
        scale_factor = 1

    if scale_factor > 1:
        image = upscale_image(image, scale_factor) 
        hook_image = upscale_image(hook_image, scale_factor)
    else:
        print("Không cần upscale")
    
    # Tính toán border dựa trên kích thước ảnh để đảm bảo tỷ lệ nhất quán
    # Sử dụng kích thước nhỏ hơn (min) để đảm bảo border không quá dày
    img_size = min(image.shape[:2])
    
    # Tính tỷ lệ so với kích thước tham chiếu (1000px)
    scale_ratio = img_size / reference_size
    
    # Áp dụng tỷ lệ cho border
    border_thickness = max(1, int(border_thickness_px * scale_ratio))
    border_padding = max(1, int(border_padding_px * scale_ratio))
    
    img_size = max(image.shape[:2])
    padding = (border_thickness + border_padding)
    hook_size = max(CONFIG['hook_size_min'], int(img_size * CONFIG['hook_size_ratio'])) 

    try:
        hook_resized = cv2.resize(hook_image, (hook_size, hook_size), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        return {"status": "error", "message": f"Lỗi khi resize ảnh móc: {e}, Kích thước ảnh móc: {hook_image.shape}"}

    expanded_image = cv2.copyMakeBorder(
        image, padding + hook_size, padding, padding, padding, 
        cv2.BORDER_CONSTANT, value=[0, 0, 0, 0]
    )
    
    image_with_hook = add_hook_to_image(expanded_image, hook_resized)
    hook_border_mask = create_inner_circle_border_mask(expanded_image, hook_resized, border_thickness)
    main_border_mask = create_main_border_mask(image_with_hook, border_thickness, border_padding)
    
    tmp_image = expanded_image.copy()
    if show_hook:
        tmp_image = image_with_hook.copy()
    
    final_result = apply_borders_to_clean_image(tmp_image, hook_border_mask, main_border_mask, border_color, show_hook)
    
    # Bước 6: Lưu ảnh kết quả
    cv2.imwrite(output_path, final_result)
    return True

def check_overlap(expanded_image, hook_y, hook_x, hook_image):
    """Kiểm tra xem móc có overlap với vật thể không"""
    height, width = expanded_image.shape[:2]
    hook_height, hook_width = hook_image.shape[:2]
    
    for y in range(hook_height):
        for x in range(hook_width):
            img_y = hook_y + y
            img_x = hook_x + x
            
            if (0 <= img_y < height and 0 <= img_x < width):
                # Nếu cả móc và vật thể đều không transparent tại pixel này
                if hook_image[y, x, 3] > 0 and expanded_image[img_y, img_x, 3] > 0:
                    return True
    return False

def get_top_point(expanded_image, center_x, min_x, max_x):
    """
    Tìm điểm cao nhất toàn bộ trước, nếu không nằm trong khoảng giới hạn thì dùng điểm cao nhất tại center_x
    
    Args:
        expanded_image: Ảnh đã expand
        center_x: Vị trí x trung tâm
        min_x: Vị trí x nhỏ nhất của vật thể
        max_x: Vị trí x lớn nhất của vật thể
    
    Returns:
        tuple: (top_x, top_y) - tọa độ điểm cao nhất
    """
    # Tính khoảng giới hạn dựa trên min_x và max_x
    width = max_x - min_x
    limit_start = min_x + int(width * CONFIG['min_limit_x'])
    limit_end = min_x + int(width * CONFIG['max_limit_x'])

    # Tìm điểm cao nhất toàn bộ
    alpha = expanded_image[:, :, 3]
    non_transparent = np.where(alpha > 0)
    
    if len(non_transparent[0]) > 0:
        # Tìm điểm cao nhất toàn bộ
        top_y = np.min(non_transparent[0])
        # Tìm tất cả x có cùng y cao nhất
        top_xs = non_transparent[1][non_transparent[0] == top_y]
        # Lấy x ở giữa các điểm cao nhất
        global_top_x = int(np.mean(top_xs))
        
        # Kiểm tra xem điểm cao nhất toàn bộ có nằm trong khoảng giới hạn không
        if limit_start <= global_top_x <= limit_end:
            return global_top_x, top_y
        else:
            # Sử dụng điểm cao nhất tại center_x
            alpha_column = expanded_image[:, center_x, 3]
            non_transparent_y = np.where(alpha_column > 0)[0]
            if len(non_transparent_y) > 0:
                center_top_y = np.min(non_transparent_y)
                return center_x, center_top_y
            else:
                # Nếu không tìm thấy tại center_x, vẫn dùng điểm cao nhất toàn bộ
                return global_top_x, top_y
    
    # Fallback: nếu không tìm thấy pixel nào
    return center_x, 0

def add_hook_to_image(expanded_image, hook_image):
    """
    Thêm ảnh móc treo vào vị trí giữa trên của vật thể
    """
    height, width = expanded_image.shape[:2]
    hook_height, hook_width = hook_image.shape[:2]
    
    # Tìm vị trí vật thể gốc trong ảnh đã expand
    alpha = expanded_image[:, :, 3]
    non_transparent = np.where(alpha > 0)
    
    if len(non_transparent[0]) == 0:
        return expanded_image
    
    # Tìm vị trí y cao nhất và thấp nhất của vật thể
    min_y = np.min(non_transparent[0])
    max_y = np.max(non_transparent[0])
    
    # Tìm tọa độ x ở vị trí y giữa của vật thể
    middle_y = (min_y + max_y) // 2
    middle_row_mask = (non_transparent[0] == middle_y)
    middle_row_x_coords = non_transparent[1][middle_row_mask]
    
    if len(middle_row_x_coords) == 0:
        # Nếu không tìm được pixel ở giữa chính xác, tìm pixel gần nhất
        nearest_y = min_y
        min_distance = abs(middle_y - min_y)
        
        for y in range(min_y, max_y + 1):
            row_mask = (non_transparent[0] == y)
            if np.any(row_mask):
                distance = abs(middle_y - y)
                if distance < min_distance:
                    min_distance = distance
                    nearest_y = y
        
        row_mask = (non_transparent[0] == nearest_y)
        middle_row_x_coords = non_transparent[1][row_mask]
    
    # Tìm trung tâm của hàng pixel giữa
    center_x = int(np.mean(middle_row_x_coords))
    
    # Tìm điểm không transparent cao nhất tại center_x
    alpha_column = expanded_image[:, center_x, 3]
    non_transparent_y = np.where(alpha_column > 0)[0]
    if len(non_transparent_y) > 0:
        top_y = np.min(non_transparent_y)
    else:
        top_y = min_y
    
    # Tìm vị trí x nhỏ nhất và lớn nhất của vật thể
    min_x = np.min(non_transparent[1])
    max_x = np.max(non_transparent[1])
    
    # Tìm tọa độ điểm cao nhất phù hợp
    hook_center_x, top_y = get_top_point(expanded_image, center_x, min_x, max_x)

    # Đặt móc ban đầu
    hook_x = hook_center_x - hook_width // 2
    hook_y = top_y - hook_height
    
    # Di chuyển móc lên trên cho đến khi không còn overlap
    while check_overlap(expanded_image, hook_y, hook_x, hook_image):
        hook_y -= 1
        if hook_y < 0:  # Đảm bảo không vượt quá canvas
            hook_y = 0
            break
    
    # Đảm bảo móc không bị cắt ra ngoài canvas theo chiều ngang
    hook_x = max(0, min(hook_x, width - hook_width))
    
    # Copy ảnh để tránh thay đổi ảnh gốc
    result = expanded_image.copy()
    
    # Thêm móc vào ảnh
    for y in range(hook_height):
        for x in range(hook_width):
            img_y = hook_y + y
            img_x = hook_x + x
            
            if (0 <= img_y < height and 0 <= img_x < width):
                hook_alpha = hook_image[y, x, 3] / 255.0
                
                if hook_alpha > 0:  # Chỉ blend pixel không trong suốt
                    # Alpha blending
                    for c in range(3):  # BGR channels
                        result[img_y, img_x, c] = (
                            hook_alpha * hook_image[y, x, c] + 
                            (1 - hook_alpha) * result[img_y, img_x, c]
                        ).astype(np.uint8)
                    
                    # Cập nhật alpha channel
                    result[img_y, img_x, 3] = max(
                        result[img_y, img_x, 3], 
                        (hook_alpha * 255).astype(np.uint8)
                    )
    
    return result

def create_inner_circle_border_mask(expanded_image, hook_image, border_thickness):
    """
    Tạo mask cho border tròn bên trong móc treo
    
    Args:
        expanded_image: Ảnh gốc đã expand (chưa có móc)
        hook_image: Ảnh móc treo gốc
        border_thickness: Độ dày của border tròn
    
    Returns:
        Mask cho border tròn (numpy array)
    """
    height, width = expanded_image.shape[:2]
    hook_height, hook_width = hook_image.shape[:2]
    
    # Tìm vị trí vật thể gốc trong ảnh đã expand
    alpha = expanded_image[:, :, 3]
    non_transparent = np.where(alpha > 0)
    
    if len(non_transparent[0]) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Tìm vị trí y cao nhất và thấp nhất của vật thể
    min_y = np.min(non_transparent[0])
    max_y = np.max(non_transparent[0])
    
    # Tìm tọa độ x ở vị trí y giữa của vật thể
    middle_y = (min_y + max_y) // 2
    middle_row_mask = (non_transparent[0] == middle_y)
    middle_row_x_coords = non_transparent[1][middle_row_mask]
    
    if len(middle_row_x_coords) == 0:
        # Nếu không tìm được pixel ở giữa chính xác, tìm pixel gần nhất
        nearest_y = min_y
        min_distance = abs(middle_y - min_y)
        
        for y in range(min_y, max_y + 1):
            row_mask = (non_transparent[0] == y)
            if np.any(row_mask):
                distance = abs(middle_y - y)
                if distance < min_distance:
                    min_distance = distance
                    nearest_y = y
        
        row_mask = (non_transparent[0] == nearest_y)
        middle_row_x_coords = non_transparent[1][row_mask]
    
    # Tìm trung tâm của hàng pixel giữa
    center_x = int(np.mean(middle_row_x_coords))
    
    # Tìm điểm không transparent cao nhất tại center_x
    alpha_column = expanded_image[:, center_x, 3]
    non_transparent_y = np.where(alpha_column > 0)[0]
    if len(non_transparent_y) > 0:
        top_y = np.min(non_transparent_y)
    else:
        # Nếu không tìm thấy pixel không transparent tại center_x,
        # sử dụng điểm cao nhất của toàn bộ vật thể
        top_y = min_y
    
    # Tìm vị trí x nhỏ nhất và lớn nhất của vật thể
    min_x = np.min(non_transparent[1])
    max_x = np.max(non_transparent[1])
    
    # Tìm tọa độ điểm cao nhất phù hợp
    hook_center_x, top_y = get_top_point(expanded_image, center_x, min_x, max_x)
    
    # Đặt móc ban đầu
    hook_x = hook_center_x - hook_width // 2
    hook_y = top_y - hook_height
    
    # Di chuyển móc lên trên cho đến khi không còn overlap
    while check_overlap(expanded_image, hook_y, hook_x, hook_image):
        hook_y -= 1
        if hook_y < 0:
            hook_y = 0
            break
    
    # Tìm tâm thực tế của móc
    hook_alpha = hook_image[:, :, 3]
    hook_non_transparent = np.where(hook_alpha > 0)
    
    if len(hook_non_transparent[0]) > 0:
        hook_center_y_relative = int(np.mean(hook_non_transparent[0]))
        hook_center_x_relative = int(np.mean(hook_non_transparent[1]))
        
        hook_center_x = hook_x + hook_center_x_relative
        hook_center_y = hook_y + hook_center_y_relative
    else:
        hook_center_x = hook_x + hook_width // 2
        hook_center_y = hook_y + hook_height // 2
    
    # Tính bán kính cho đường tròn bên trong
    inner_radius = int(min(hook_width, hook_height) * CONFIG['inner_radius_ratio'])
    
    # Tạo mask cho border tròn
    border_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(border_mask, (hook_center_x, hook_center_y), inner_radius, 255, border_thickness)
    
    # Làm mịn
    border_mask = cv2.GaussianBlur(border_mask, (3, 3), 0)
    
    return border_mask

def create_main_border_mask(image_with_hook, border_thickness, border_padding):
    """
    Tạo mask cho border chính của toàn bộ vật thể
    
    Args:
        image_with_hook: Ảnh đã có móc treo
    
    Returns:
        Mask cho border chính
    """
    # Tạo mask từ alpha channel
    alpha = image_with_hook[:, :, 3]
    mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)[1]
    
    # Border width: scale theo kích thước ảnh
    border_width = border_thickness + border_padding

    # Kernel tròn để tạo hiệu ứng bo tròn
    kernel_size = (border_padding + border_thickness // 2) * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Tạo mask dilated với kernel tròn
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Tìm contours và tạo border
    contours, _ = cv2.findContours(
        cv2.threshold(dilated_mask, 127, 255, cv2.THRESH_BINARY)[1], 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Tạo mask cho border
    border_layer = np.zeros_like(dilated_mask)
    cv2.drawContours(border_layer, contours, -1, 255, thickness=border_thickness)
    
    # Loại bỏ phần bên trong để chỉ giữ lại border
    only_border = border_layer.copy()
    if (border_padding > 0):
        erode_size = border_padding
        erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
        inner_mask = cv2.erode(mask, erode_kernel, iterations=1)
        only_border = cv2.bitwise_and(border_layer, cv2.bitwise_not(inner_mask))
    
    # Làm mịn border để có cạnh smooth
    smooth_kernel_size = border_thickness
    if smooth_kernel_size % 2 == 0:
        smooth_kernel_size += 1
    only_border = cv2.GaussianBlur(only_border, (smooth_kernel_size, smooth_kernel_size), 0)
    
    return only_border

def apply_borders_to_clean_image(clean_image, hook_border_mask, main_border_mask, border_color, show_hook):
    """
    Áp dụng cả 2 border lên ảnh gốc đã expand (không có móc)
    
    Args:
        clean_image: Ảnh gốc đã expand (không có móc treo)
        hook_border_mask: Mask cho border tròn bên trong móc
        main_border_mask: Mask cho border chính
        show_hook: Hiển thị border móc treo hay không
    
    Returns:
        Ảnh với cả 2 border được áp dụng
    """
    result = clean_image.copy()

    bgr = hex_to_bgr(border_color)  
    bd_color = np.array(bgr, dtype=np.uint8)

    # Áp dụng border chính
    main_border_pixels = main_border_mask > 10
    if np.any(main_border_pixels):
        alpha_values = main_border_mask[main_border_pixels] / 255.0
        
        for c in range(3):  # BGR channels
            result[main_border_pixels, c] = (
                alpha_values * bd_color[c] + 
                (1 - alpha_values) * result[main_border_pixels, c]
            ).astype(np.uint8)
        
        # Cập nhật alpha channel
        new_alpha = (alpha_values * 255).astype(np.uint8)
        result[main_border_pixels, 3] = np.maximum(result[main_border_pixels, 3], new_alpha)
    
    # Áp dụng border tròn (móc)
    if not show_hook:
        hook_border_pixels = hook_border_mask > 10
        if np.any(hook_border_pixels):
            alpha_values = hook_border_mask[hook_border_pixels] / 255.0
            
            for c in range(3):  # BGR channels
                result[hook_border_pixels, c] = (
                    alpha_values * bd_color[c] + 
                    (1 - alpha_values) * result[hook_border_pixels, c]
                ).astype(np.uint8)
            
            # Cập nhật alpha channel
            new_alpha = (alpha_values * 255).astype(np.uint8)
            result[hook_border_pixels, 3] = np.maximum(result[hook_border_pixels, 3], new_alpha)
    
    return result

def process_all_images(input_folder='input', output_folder='output', hook_image_path='circle.png'):
    """
    Xử lý tất cả ảnh trong folder input và lưu vào folder output
    
    Args:
        input_folder (str): Tên folder chứa ảnh đầu vào
        output_folder (str): Tên folder để lưu ảnh kết quả
        hook_image_path (str): Đường dẫn đến ảnh móc treo
    """
    # Kiểm tra ảnh móc treo có tồn tại không
    if not os.path.exists(hook_image_path):
        print(f"Không tìm thấy ảnh móc treo: {hook_image_path}")
        return
    
    # Tạo folder output nếu chưa tồn tại
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Các định dạng ảnh được hỗ trợ
    supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    if not os.path.exists(input_folder):
        print(f"Folder '{input_folder}' không tồn tại!")
        return
    
    # Lấy danh sách file ảnh
    image_files = []
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            if ext.lower() in supported_formats:
                image_files.append(filename)
    
    if not image_files:
        print(f"Không tìm thấy ảnh nào trong folder '{input_folder}'!")
        return
    
    print(f"Tìm thấy {len(image_files)} ảnh để xử lý...")
    print(f"Sử dụng ảnh móc treo: {hook_image_path}")
    
    processed_count = 0
    failed_count = 0
    
    for image_file in image_files:
        input_file_path = os.path.join(input_folder, image_file)
        filename_without_ext, ext = os.path.splitext(image_file)
        output_filename = f"{filename_without_ext}_with_border{ext}"
        output_file_path = os.path.join(output_folder, output_filename)
        
        print(f"Đang xử lý: {image_file}")
        
        # Sử dụng giá trị pixel tương đối với ảnh 1000x1000px
        # border_thickness_px: 5px trên ảnh 1000x1000px
        # border_padding_px: 8px trên ảnh 1000x1000px
        if processed_image(input_file_path, output_file_path, 5, 8, "#f3202c", True):
            processed_count += 1
            print(f"  ✓ Đã lưu: {output_filename}")
        else:
            failed_count += 1
            print(f"  ✗ Lỗi khi xử lý: {image_file}")
    
    print(f"\nHoàn thành!")
    print(f"- Đã xử lý thành công: {processed_count} ảnh")
    print(f"- Lỗi: {failed_count} ảnh")
    print(f"- Kết quả được lưu trong folder '{output_folder}'")

if __name__ == "__main__":
    # Chạy xử lý batch cho tất cả ảnh
    process_all_images()