# import tensorflow as tf
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from tensorflow.keras.metrics import MeanSquaredError
# import streamlit as st
# import numpy as np
# from PIL import Image, ImageDraw

# # Định nghĩa hàm MSE đơn giản
# def mse(y_true, y_pred):
#     return tf.reduce_mean(tf.square(y_true - y_pred))

# # Định nghĩa custom objects
# custom_objects = {
#     'SparseCategoricalCrossentropy': SparseCategoricalCrossentropy,
#     'mse': mse
# }

# # Load mô hình Faster R-CNN từ file đã huấn luyện
# model_path = 'faster_rcnn_new_model.h5'
# model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# # Hàm nhận diện và vẽ bounding box
# def detect_and_draw(image):
#     img = np.array(image)
#     img = img[np.newaxis, ...]  # Thêm một chiều để phù hợp với input của mô hình
#     rpn_cls_pred, rpn_reg_pred = model.predict(img)
    
#     # Xử lý rpn_cls_pred và rpn_reg_pred để nhận diện và tính toán bounding box
    
#     # Vẽ bounding box lên ảnh
#     draw = ImageDraw.Draw(image)
#     # Ví dụ vẽ bounding box ở đây
#     draw.rectangle([(50, 50), (100, 100)], outline='red', width=2)
    
#     return image

# # Xây dựng giao diện người dùng trong Streamlit
# st.title('Faster R-CNN Plant Disease Detection')

# uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
    
#     if st.button('Detect'):
#         # Thực hiện nhận diện và vẽ bounding box
#         result_image = detect_and_draw(image)
        
#         # Hiển thị ảnh kết quả
#         st.image(result_image, caption='Result Image.', use_column_width=True)
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load mô hình
model = tf.keras.models.load_model('faster_rcnn_model.h5', compile=False)

# Hàm dự đoán
def predict(image):
    # Chuyển đổi ảnh sang định dạng phù hợp với mô hình
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image, (256, 256))
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # Dự đoán
    rpn_cls_pred, rpn_reg_pred, cls_output, reg_output = model.predict(image_array)
    
    # Xử lý output để lấy bounding box và nhãn
    cls_index = np.argmax(cls_output)
    label = cls_index  # hoặc ánh xạ chỉ số sang tên bệnh
    bbox = reg_output[0]
    
    return label, bbox

# Hàm vẽ bounding box lên ảnh
def draw_bounding_box(image, bbox):
    (h, w) = image.shape[:2]
    startX, startY, endX, endY = bbox
    startX = int(startX * w)
    startY = int(startY * h)
    endX = int(endX * w)
    endY = int(endY * h)
    
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
    return image

# Ứng dụng Streamlit
st.title("Plant Disease Detection with Bounding Box")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button("Predict"):
        label, bbox = predict(image)
        st.write(f"Predicted label: {label}")
        
        image_with_bbox = draw_bounding_box(image, bbox)
        st.image(image_with_bbox, caption='Image with Bounding Box', use_column_width=True)

