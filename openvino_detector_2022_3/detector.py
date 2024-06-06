
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from openvino.inference_engine import IECore

# Initialize OpenVINO core and read the model
ie = IECore()
detection_model_xml = "C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
detection_model_bin = os.path.splitext(detection_model_xml)[0] + ".bin"
detection_net = ie.read_network(model=detection_model_xml, weights=detection_model_bin)
device = "CPU"
exec_net = ie.load_network(network=detection_net, device_name=device)

# Video path
video_path = 'openvino_detector_2022_3/videos/videos/person_media.mp4'

# Open the video file
cap = cv2.VideoCapture(0)

# Initialize variables for counting people
total_people = 0

# Obtain model input dimensions
input_layer = next(iter(detection_net.input_info))
output_layer = next(iter(detection_net.outputs))
N, C, H, W = detection_net.input_info[input_layer].input_data.shape

# Define the frame skip interval
frame_skip = 5  # Display every 5th frame

# Prepare Matplotlib figure for displaying the video
fig, ax = plt.subplots(figsize=(10, 6))

# Initialize frame count
frame_count = 0

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Increment frame count
    frame_count += 1

    # Skip frames based on frame_skip interval
    if frame_count % frame_skip != 0:
        continue

    # Preprocess the frame
    resized_image = cv2.resize(src=frame, dsize=(W, H))
    input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0)

    # Perform inference
    result = exec_net.infer(inputs={input_layer: input_data})

    # Process detection results
    frame_height, frame_width = frame.shape[:2]
    num_people = 0
    for detection in result[output_layer][0][0]:
        conf = float(detection[2])
        if conf > 0.76:
            xmin = int(detection[3] * frame_width)
            ymin = int(detection[4] * frame_height)
            xmax = int(detection[5] * frame_width)
            ymax = int(detection[6] * frame_height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
            cv2.putText(frame, "person", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            num_people += 1
            total_people += 1

    # Display the frame with count using Matplotlib
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f'Person Detection Demo - Number of People: {num_people}')
    ax.axis('off')
    plt.pause(0.001)

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Total number of people detected:", total_people)








# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from openvino.runtime import Core
# from IPython.display import display, clear_output

# # Initialize OpenVINO core and read the model
# core = Core()
# # detection_model_xml = "model_2022_3/person-detection-retail-0013.xml" 
# detection_model_xml = "C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
#  # Update the path accordingly
# detection_model = core.read_model(model=detection_model_xml)
# device = "CPU"
# compiled_model = core.compile_model(model=detection_model, device_name=device)

# # Image path
# image_path = 'C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/videos/videos/image8.jpg'

# # Read the image
# frame = cv2.imread(image_path)
# if frame is None:
#     print("Error: Could not read the image.")
#     exit()

# #

# # Obtain model input dimensions
# input_layer = compiled_model.input(0)
# output_layer = compiled_model.output(0)
# N, C, H, W = input_layer.shape

# # Preprocess the frame
# resized_image = cv2.resize(src=frame, dsize=(W, H))
# input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)

# # Perform inference
# request = compiled_model.create_infer_request()
# request.infer(inputs={input_layer.any_name: input_data})
# result = request.get_output_tensor(output_layer.index).data

# # Process detection results
# frame_height, frame_width = frame.shape[:2]
# for detection in result[0][0]:
#     conf = float(detection[2])
#     if conf > 0.76:
#         xmin = int(detection[3] * frame_width)
#         ymin = int(detection[4] * frame_height)
#         xmax = int(detection[5] * frame_width)
#         ymax = int(detection[6] * frame_height)
#         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
#         cv2.putText(frame, "person", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

# # Display the frame
# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.title('Person Detection Demo')
# plt.axis('off')
# plt.show()













# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from openvino.runtime import Core
# from IPython.display import display, clear_output

# # Initialize OpenVINO core and read the model
# core = Core()
# # detection_model_xml = "model_2022_3/person-detection-retail-0013.xml" 
# detection_model_xml = "C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
#  # Update the path accordingly
# detection_model = core.read_model(model=detection_model_xml)
# device = "CPU"
# compiled_model = core.compile_model(model=detection_model, device_name=device)

# # Image path
# image_path = 'C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/videos/videos/image3.jpg'

# # Read the image
# frame = cv2.imread(image_path)
# if frame is None:
#     print("Error: Could not read the image.")
#     exit()

# # Obtain model input dimensions
# input_layer = compiled_model.input(0)
# output_layer = compiled_model.output(0)
# N, C, H, W = input_layer.shape

# # Preprocess the frame
# resized_image = cv2.resize(src=frame, dsize=(W, H))
# input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)

# # Perform inference
# request = compiled_model.create_infer_request()
# request.infer(inputs={input_layer.any_name: input_data})
# result = request.get_output_tensor(output_layer.index).data

# # Process detection results
# frame_height, frame_width = frame.shape[:2]
# num_people = 0
# for detection in result[0][0]:
#     conf = float(detection[2])
#     if conf > 0.76:
#         xmin = int(detection[3] * frame_width)
#         ymin = int(detection[4] * frame_height)
#         xmax = int(detection[5] * frame_width)
#         ymax = int(detection[6] * frame_height)
#         cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
#         cv2.putText(frame, "person", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#         num_people += 1

# # Display the frame with count
# plt.figure(figsize=(10, 6))
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.title(f'Person Detection Demo - Number of People: {num_people}')
# plt.axis('off')
# plt.show()











# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from openvino.runtime import Core

# # Initialize OpenVINO core and read the model
# core = Core()
# detection_model_xml = "C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
# detection_model = core.read_model(model=detection_model_xml)
# device = "CPU"
# compiled_model = core.compile_model(model=detection_model, device_name=device)

# # Video path
# video_path = 'openvino_detector_2022_3/videos/videos/person.mp4'

# # Open the video file
# cap = cv2.VideoCapture(video_path)

# # Initialize variables for counting people
# total_people = 0

# # Obtain model input dimensions
# input_layer = compiled_model.input(0)
# output_layer = compiled_model.output(0)
# N, C, H, W = input_layer.shape

# while cap.isOpened():
#     # Read a frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Preprocess the frame
#     resized_image = cv2.resize(src=frame, dsize=(W, H))
#     input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)

#     # Perform inference
#     request = compiled_model.create_infer_request()
#     request.infer(inputs={input_layer.any_name: input_data})
#     result = request.get_output_tensor(output_layer.index).data

#     # Process detection results
#     frame_height, frame_width = frame.shape[:2]
#     num_people = 0
#     for detection in result[0][0]:
#         conf = float(detection[2])
#         if conf > 0.76:
#             xmin = int(detection[3] * frame_width)
#             ymin = int(detection[4] * frame_height)
#             xmax = int(detection[5] * frame_width)
#             ymax = int(detection[6] * frame_height)
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
#             cv2.putText(frame, "person", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#             num_people += 1
#             total_people += 1

#     # Display the frame with count
#     cv2.imshow('Frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# print("Total number of people detected:", total_people)











# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from openvino.runtime import Core

# # Initialize OpenVINO core and read the model
# core = Core()


# detection_model_xml = "C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
# detection_model = core.read_model(model=detection_model_xml)
# device = "CPU"
# compiled_model = core.compile_model(model=detection_model, device_name=device)

# # Video path
# video_path = 'openvino_detector_2022_3/videos/videos/person_media.mp4'

# # Open the video file
# cap = cv2.VideoCapture(video_path)

# # Initialize variables for counting people
# total_people = 0

# # Obtain model input dimensions
# input_layer = compiled_model.input(0)
# output_layer = compiled_model.output(0)
# N, C, H, W = input_layer.shape

# # Prepare Matplotlib figure for displaying the video
# fig, ax = plt.subplots(figsize=(10, 6))

# while cap.isOpened():
#     # Read a frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Preprocess the frame
#     resized_image = cv2.resize(src=frame, dsize=(W, H))
#     input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)

#     # Perform inference
#     request = compiled_model.create_infer_request()
#     request.infer(inputs={input_layer.any_name: input_data})
#     result = request.get_output_tensor(output_layer.index).data

#     # Process detection results
#     frame_height, frame_width = frame.shape[:2]
#     num_people = 0
#     for detection in result[0][0]:
#         conf = float(detection[2])
#         if conf > 0.76:
#             xmin = int(detection[3] * frame_width)
#             ymin = int(detection[4] * frame_height)
#             xmax = int(detection[5] * frame_width)
#             ymax = int(detection[6] * frame_height)
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
#             cv2.putText(frame, "person", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#             num_people += 1
#             total_people += 1

#     # Display the frame with count using Matplotlib
#     ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     ax.set_title(f'Person Detection Demo - Number of People: {num_people}')
#     ax.axis('off')
#     plt.pause(0.001)
  

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# print("Total number of people detected:", total_people)













# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from openvino.runtime import Core

# # Initialize OpenVINO core and read the model
# core = Core()
# detection_model_xml = "C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
# detection_model = core.read_model(model=detection_model_xml)
# device = "CPU"
# compiled_model = core.compile_model(model=detection_model, device_name=device)

# # Video path
# video_path = 'openvino_detector_2022_3/videos/videos/person_media.mp4'

# # Open the video file
# cap = cv2.VideoCapture(0)

# # Initialize variables for counting people
# total_people = 0

# # Obtain model input dimensions
# input_layer = compiled_model.input(0)
# output_layer = compiled_model.output(0)
# N, C, H, W = input_layer.shape

# # Define the frame skip interval
# frame_skip = 5  # Display every 5th frame

# # Prepare Matplotlib figure for displaying the video
# fig, ax = plt.subplots(figsize=(10, 6))

# # Initialize frame count
# frame_count = 0

# while cap.isOpened():
#     # Read a frame from the video
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Increment frame count
#     frame_count += 1

#     # Skip frames based on frame_skip interval
#     if frame_count % frame_skip != 0:
#         continue

#     # Preprocess the frame
#     resized_image = cv2.resize(src=frame, dsize=(W, H))
#     input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)

#     # Perform inference
#     request = compiled_model.create_infer_request()
#     request.infer(inputs={input_layer.any_name: input_data})
#     result = request.get_output_tensor(output_layer.index).data

#     # Process detection results
#     frame_height, frame_width = frame.shape[:2]
#     num_people = 0
#     for detection in result[0][0]:
#         conf = float(detection[2])
#         if conf > 0.76:
#             xmin = int(detection[3] * frame_width)
#             ymin = int(detection[4] * frame_height)
#             xmax = int(detection[5] * frame_width)
#             ymax = int(detection[6] * frame_height)
#             cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
#             cv2.putText(frame, "person", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#             num_people += 1
#             total_people += 1

#     # Display the frame with count using Matplotlib
#     ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     ax.set_title(f'Person Detection Demo - Number of People: {num_people}')
#     ax.axis('off')
#     plt.pause(0.001)

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# print("Total number of people detected:", total_people)


















# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# from threading import Thread
# from openvino.runtime import Core

# # Initialize OpenVINO core and read the model
# core = Core()
# detection_model_xml = "C:/Users/DELL/Downloads/openvino_detector_2022_3/openvino_detector_2022_3/model_2022_3/person-detection-retail-0013.xml"
# detection_model = core.read_model(model=detection_model_xml)
# device = "CPU"
# compiled_model = core.compile_model(model=detection_model, device_name=device)

# # Video path
# video_path = 'openvino_detector_2022_3/videos/videos/person_media.mp4'

# # Open the video file
# cap = cv2.VideoCapture(video_path)

# # Initialize variables for counting people
# total_people = 0

# # Obtain model input dimensions
# input_layer = compiled_model.input(0)
# output_layer = compiled_model.output(0)
# N, C, H, W = input_layer.shape

# # Prepare Matplotlib figure for displaying the video
# fig, ax = plt.subplots(figsize=(10, 6))

# def process_frames():
#     global total_people
#     while cap.isOpened():
#         # Read a frame from the video
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Preprocess the frame
#         resized_image = cv2.resize(src=frame, dsize=(W, H))
#         input_data = np.expand_dims(np.transpose(resized_image, (2, 0, 1)), 0).astype(np.float32)

#         # Perform inference
#         request = compiled_model.create_infer_request()
#         request.infer(inputs={input_layer.any_name: input_data})
#         result = request.get_output_tensor(output_layer.index).data

#         # Process detection results
#         frame_height, frame_width = frame.shape[:2]
#         num_people = 0
#         for detection in result[0][0]:
#             conf = float(detection[2])
#             if conf > 0.76:
#                 xmin = int(detection[3] * frame_width)
#                 ymin = int(detection[4] * frame_height)
#                 xmax = int(detection[5] * frame_width)
#                 ymax = int(detection[6] * frame_height)
#                 cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
#                 cv2.putText(frame, "person", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
#                 num_people += 1
#                 total_people += 1

#         # Display the frame with count using Matplotlib
#         ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         ax.set_title(f'Person Detection Demo - Number of People: {num_people}')
#         ax.axis('off')
#         plt.pause(0.001)

# # Start the frame processing thread
# frame_thread = Thread(target=process_frames)
# frame_thread.start()

# # Wait for the frame processing thread to finish
# frame_thread.join()

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# print("Total number of people detected:", total_people)