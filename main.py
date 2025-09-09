import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from os.path import join, dirname, join
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.checkbox import CheckBox
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image as KivyImage
from kivy.uix.video import Video
from kivy.uix.videoplayer import VideoPlayer
from kivy.uix.image import Image
from kivy.uix.popup import Popup
from kivy.properties import ObjectProperty
from kivy.graphics.texture import Texture
from kivy.uix.spinner import Spinner
from kivy.uix.filechooser import FileChooserListView
from kivy.core.window import Window
import coord_metrics2 as cm
import sys
import io
import cv2
import mediapipe as mp
import csv
from kivy.clock import Clock
import os 
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from kivy.graphics import Color, Line
from kivy.core.window import Window
from scipy.signal import filtfilt, butter
import matplotlib
matplotlib.use('Agg')

def butter_lowpass(cutoff, order):
    # Returns the filter coefficients for a Butterworth low-pass filter
    b, a = butter(order, cutoff, btype='low', analog=False)
    return b, a

Clock.max_iteration = 50  # Default is 20
from kivy.utils import platform

if platform == 'android':
    from android.permissions import request_permissions, Permission
else:
    def request_permissions(*args, **kwargs):
        pass
    class Permission:
        READ_EXTERNAL_STORAGE = ''
        WRITE_EXTERNAL_STORAGE = ''

# get any files into images directory
ROOT = join(dirname(__file__), 'data/')

# Joints landmarks
left_landmarks = ['Left Wrist', 'Left Elbow', 'Left Shoulder', 'Left Hip', 'Left Knee', 'Left Ankle']
right_landmarks = ['Right Wrist', 'Right Elbow', 'Right Shoulder', 'Right Hip', 'Right Knee', 'Right Ankle']
        

markers_right = [('RIGHT_ELBOW', 'RIGHT_WRIST', 'RIGHT_INDEX'), ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
                ('RIGHT_HIP', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_KNEE', 'RIGHT_HIP', 'RIGHT_SHOULDER'),
                ('RIGHT_ANKLE', 'RIGHT_KNEE', 'RIGHT_HIP'), ('RIGHT_FOOT_INDEX', 'RIGHT_ANKLE', 'RIGHT_KNEE')]

markers_left = [('LEFT_ELBOW', 'LEFT_WRIST', 'LEFT_INDEX'), ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
            ('LEFT_HIP', 'LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_KNEE', 'LEFT_HIP', 'LEFT_SHOULDER'),
            ('LEFT_ANKLE', 'LEFT_KNEE', 'LEFT_HIP'), ('LEFT_FOOT_INDEX', 'LEFT_ANKLE', 'LEFT_KNEE')]



def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def choose_possible_landmarks(file_ref, file_comp):
    
    # Replace the actual extension of file_ref to .csv
    if not file_ref.endswith('.csv'):
        file_ref = os.path.join(os.path.dirname(file_ref), os.path.splitext(os.path.basename(file_ref))[0] + '.csv')
    if not file_comp.endswith('.csv'):
        file_comp = os.path.join(os.path.dirname(file_comp), os.path.splitext(os.path.basename(file_comp))[0] + '.csv')
        
    # Open both files as dataframes
    df_ref = pd.read_csv(file_ref)
    df_comp = pd.read_csv(file_comp)
    
    # Merge the two dataframes on the 'landmark' column
    merged_df = pd.merge(df_ref, df_comp, on='landmark', suffixes=('_ref', '_comp'))

    # Count the total number of lines in the merged dataframe
    total_lines = len(merged_df)
    print(f"Total number of lines in the merged dataframe: {total_lines}")
            
    # Extract unique markers from markers_left
    m_left = list(set(marker for triplet in markers_left for marker in triplet))
    
    # Extract unique markers from markers_right
    m_right = list(set(marker for triplet in markers_right for marker in triplet))
    print(m_right)

    # Filter landmarks based on their presence in more than 1/24 of the lines
    threshold = total_lines / 1000
    left_markers = [landmark for landmark in m_left if merged_df['landmark'].value_counts().get(landmark, 0) > threshold]
    right_markers = [landmark for landmark in m_right if merged_df['landmark'].value_counts().get(landmark, 0) > threshold]
    
    return left_markers, right_markers
            
def write_landmarks_to_csv(mp_pose, landmarks, frame_number, csv_data):
    pose = mp_pose.Pose()
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
    print("\n")
    
def show_results(list_of_fig, summary=None): 
    
    # Create a layout to hold both images
    result_layout = BoxLayout(orientation='vertical')
    
    for fig in list_of_fig:
        texture = figure_to_texture(fig)
        kivy_image = KivyImage()
        kivy_image.texture = texture
        result_layout.add_widget(kivy_image)
    if summary:
        summary_label = Label(text=summary, size_hint=(1, 0.2), halign='center', valign='middle', text_size=(Window.width * 0.8, None))
        result_layout.add_widget(summary_label)   
    # Create popup
    popup = Popup(title='Results', content=result_layout, size_hint=(0.9, 0.9))
    popup.open()

    
# Filter parameters
cutoff_frequency = 3.0  # Desired cutoff frequency of the filter (Hz)
sampling_rate = 24.0    # Sampling rate of the signal (Hz)
nyquist_frequency = 0.5 * sampling_rate  # Nyquist frequency

# Normalize the cutoff frequency with respect to the Nyquist frequency
normalized_cutoff = cutoff_frequency / nyquist_frequency

# Design a low-pass Butterworth filter
order = 5  # Filter order

def low_pass_filter(order, normalized_cutoff,signal_data):
    b, a = butter_lowpass(normalized_cutoff, order)
    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal_data)
    return filtered_signal


def angle_between_points(A, B, C):
    # Convert the points to NumPy arrays
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    
    # Calculate vectors BA and BC
    BA = A - B
    BC = C - B
    
    # Dot product of BA and BC
    dot_product = np.dot(BA, BC)
    
    # Magnitudes of BA and BC
    magnitude_BA = np.linalg.norm(BA)
    magnitude_BC = np.linalg.norm(BC)
    
    # Cosine of the angle between BA and BC
    cos_theta = dot_product / (magnitude_BA * magnitude_BC)
    
    # Clip the value to avoid any potential floating point errors leading to values out of [-1, 1]
    # cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Angle in radians
    theta = np.arccos(cos_theta)
    
    # Convert the angle from radians to degrees if desired
    angle_in_degrees = np.degrees(theta)
    
    return theta, angle_in_degrees  # Returns angle in radians and degrees



def figure_to_texture(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    pil_image = PILImage.open(buf).convert('RGBA')
    width, height = pil_image.size
    texture = Texture.create(size=(width, height))
    texture.blit_buffer(pil_image.tobytes(), colorfmt='rgba', bufferfmt='ubyte')
    texture.flip_vertical()
    return texture



kivy.require('2.0.0')
# Set the app to use the full size of the phone screen
Window.clearcolor = (1, 1, 1, 1)  # White
Window.size = (800, 600)
class Header(BoxLayout):
    def __init__(self, **kwargs):
        self._aspect_ratio=None
        super(Header, self).__init__(orientation='horizontal', size_hint=(1, 0.1), **kwargs)
        self.spacing = 10
        self.padding = 10

        # Add an image to the header
        self.logo = Image(source=join(dirname(__file__), "isir-trans.png"), size_hint=(0.1, 1))
        self.add_widget(self.logo)

        # Add a label with the app name
        self.app_name = Label(text="CoordiMotion", font_size='20sp', bold=True, size_hint=(0.8, 1), halign='center', valign='middle', color=(0, 0, 0, 1))
        self.app_name.bind(size=self.app_name.setter('text_size'))
        self.add_widget(self.app_name)
        
        # Add a help button
        self.help_button = Button(text="?", size_hint=(0.1, 1))
        self.help_button.bind(on_press=self.show_help)
        self.add_widget(self.help_button)

       
        # Add a home button
        self.home_button = Button(size_hint=(0.2, 1))
        self.home_button.background_normal = join(dirname(__file__), "home.png")
        self.home_button.background_down = join(dirname(__file__), "home.png")
        self.home_button.bind(size=self._keep_aspect_ratio)

       
        self.home_button.bind(on_press=self.go_home)
        self.add_widget(self.home_button)
        
    def show_help(self, instance):
            # Logic to display help information
            help_text = (
                "Welcome to CoordiMotion!\n\n"
                "1. Use the home button to return to the main screen.\n"
                "2. Select reference and comparison videos to analyze.\n"
                "3. Extract angles and compare joint movements.\n"
                "4. Use the '+' button to view detailed results.\n"
                "5. Explore more metrics for advanced analysis.\n\n"
                "For further assistance, contact support."
            )
            popup = Popup(title="Help", content=Label(text=help_text, halign='center', valign='middle'), size_hint=(0.8, 0.8))
            popup.open()
        

        
    def _keep_aspect_ratio(self, instance, size):
        if self._aspect_ratio is None and instance.background_normal:
            image = PILImage.open(instance.background_normal)
            self._aspect_ratio = image.width / image.height
        if self._aspect_ratio:
            def set_width(dt):
                instance.width = instance.height * self._aspect_ratio
            Clock.schedule_once(set_width, 0)
            

    def go_home(self, instance):
        # Logic to navigate to the home view
        app = App.get_running_app()
        app.root.clear_widgets()
        header = Header()
        app.root.add_widget(header)
        app.root.add_widget(HomeView())
        
        
class HomeView(BoxLayout):
    reference_file = ObjectProperty(None)
    comparison_file = ObjectProperty(None)

    
    def __init__(self):
        super(HomeView, self).__init__(orientation='vertical', spacing=10, padding=10)
    
    
    def open_file_picker_ref(self):
        
        if platform == 'android':
            start_path = '/storage/emulated/0/DCIM/Camera'
        else :
            # Try to open the user's home directory if possible, otherwise fallback to root
            home_dir = os.path.expanduser('~')
            if os.path.exists(home_dir):
                start_path = home_dir
            else:
                start_path = '/'
        file_chooser = FileChooserListView(path=start_path)
        popup = Popup(title="Select a Video for Reference", content=file_chooser, size_hint=(0.9, 0.9))

        def on_file_selected(instance, selection, touch):
            if selection:
                self.reference_file = selection[0]
                print(f"Selected file: {self.reference_file}")
                self.file_label_ref.text = "Selected file: " + self.reference_file
                popup.dismiss()
                self.process_video(self.reference_file)

        file_chooser.bind(on_submit=on_file_selected)
        popup.open()
        
    def open_file_picker_comp(self):
        if platform == 'android':
            start_path = '/storage/emulated/0/DCIM/Camera'
        else:
            # Try to open the user's home directory if possible, otherwise fallback to root
            home_dir = os.path.expanduser('~')
            if os.path.exists(home_dir):
                start_path = home_dir
            else:
                start_path = '/'
        file_chooser = FileChooserListView(path=start_path)
        popup = Popup(title="Select a Video to Compare", content=file_chooser, size_hint=(0.9, 0.9))

        def on_file_selected(instance, selection, touch):
            if selection:
                self.comparison_file = selection[0]
                print(f"Selected file: {self.comparison_file}")
                self.file_label_comp.text = "Selected file: " + self.comparison_file
                popup.dismiss()
                self.process_video(self.comparison_file)

        file_chooser.bind(on_submit=on_file_selected)
        popup.open()
        
    def process_video(self, file):
        
        print(file)
        if file : 
            # Initialize MediaPipe Pose and Drawing utilities
            mp_pose = mp.solutions.pose
            mp_drawing = mp.solutions.drawing_utils
            
            # Open the video file
            cap = cv2.VideoCapture(file)
            frame_number = 0
            csv_data = []
            pose = mp_pose.Pose()

            # Create an Image widget to display the video frames
            video_popup = Popup(title="Video Reference", size_hint=(0.9, 0.9))
            video_layout = BoxLayout(orientation='vertical')
            video_image = Image()
            video_layout.add_widget(video_image)
            video_popup.content = video_layout
            video_popup.open()

    
            def on_video_processing_complete():
                # Save the CSV data to a file
                # Get the directory of the file and prepend it to output_csv
                file_dir = os.path.dirname(file)
                output_csv = os.path.join(file_dir, file.split('/')[-1].split('.')[0] + '.csv')
                print('Save file ' + output_csv)
                with open(output_csv, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['frame_number', 'landmark', 'x', 'y', 'z'])
                    csv_writer.writerows(csv_data)

            # Schedule the function to run after video processing is complete
            def update_frame(dt):
                nonlocal frame_number
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    video_popup.dismiss()
                    Clock.unschedule(update_frame)
                    on_video_processing_complete()  # Call the function here
                    return

                # Rotate the frame 90 degrees clockwise
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # Process the frame with MediaPipe Pose
                result = pose.process(frame)

                # Draw the pose landmarks on the frame
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        result.pose_landmarks, 
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=8),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5)
                    )

                    # Add the landmark coordinates to the list and print them
                    write_landmarks_to_csv(mp_pose, result.pose_landmarks.landmark, frame_number, csv_data)

                # Update the Image widget with the current frame
                buf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()
                texture = video_image.texture
                if not texture or texture.size != (frame.shape[1], frame.shape[0]):
                    video_image.texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
                    video_image.texture.flip_horizontal()
                video_image.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                video_image.canvas.ask_update()

                frame_number += 1

            # Schedule the frame updates
            Clock.schedule_interval(update_frame, 1.0 / 30.0)  # Assuming 30 FPS

        else:
            popup = Popup(title="Error", content=Label(text="Please select a reference video."), size_hint=(0.6, 0.4))
            popup.open()
        pass
    
    def extract_angles(self):
        
        if not self.reference_file or not self.comparison_file:
            popup = Popup(title="Error", content=Label(text="Please select both reference and comparison videos."), size_hint=(0.6, 0.4))
            popup.open()
            return
        print('Extracting angles...')
        # Create a popup with a grid layout for checkboxes
    

        # Dictionary to store checkbox states
        self.selected_landmarks = {}
        
        left_m, right_m = choose_possible_landmarks(self.reference_file, self.comparison_file)
        print(f"Left markers: {left_m}")
       
        
        # Filter landmarks based on triplets in left_markers and right_markers
        selected_left_landmarks = []
        selected_right_landmarks = []

        for i, triplet in enumerate(markers_left):
            if all(marker in left_m for marker in triplet):
                selected_left_landmarks.append(left_landmarks[i])

        for i, triplet in enumerate(markers_right):
            if all(marker in right_m for marker in triplet):
                selected_right_landmarks.append(right_landmarks[i])

        print(f"Selected Left Landmarks: {selected_left_landmarks}")
        print(f"Selected Right Landmarks: {selected_right_landmarks}")
    
        if not selected_left_landmarks or not selected_right_landmarks:
            popup = Popup(title="Error", content=Label(text="No landmarks available for selection."), size_hint=(0.6, 0.4))
            popup.open()
            return
        popup_layout = GridLayout(cols=2, spacing=10, padding=10, size_hint_y=None)
        popup_layout.bind(minimum_height=popup_layout.setter('height'))
        scroll_view = ScrollView(size_hint=(1, 0.8))
        scroll_view.add_widget(popup_layout)

        # Merge left and right landmarks into a single list
        all_landmarks = selected_left_landmarks + selected_right_landmarks

        # Add checkboxes for all landmarks
        for landmark in all_landmarks:
            checkbox = CheckBox()
            popup_layout.add_widget(Label(text=landmark, size_hint_y=None, height=40))
            popup_layout.add_widget(checkbox)
            self.selected_landmarks[landmark] = checkbox

        # Create a popup
        # Add an OK button at the bottom of the popup
        ok_button = Button(text="OK", size_hint=(1, 0.2))
        
        def on_ok_button_press(instance):
            popup.dismiss()
            selected_landmarks = [landmark for landmark, checkbox in self.selected_landmarks.items() if checkbox.active]
            print(f"Selected landmarks: {selected_landmarks}")
            self.extract_single_joint(selected_landmarks)  # Pass the selected landmarks to the function
            
        # Add a spinner to choose a number between 1 and 10
        spinner_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.2), spacing=10, padding=10)
        spinner_label = Label(text="Number of Dof of the task:", size_hint=(0.6, 1))
        spinner = Spinner(
            text="2",
            values=[str(i) for i in range(1, 11)],
            size_hint=(0.4, 1)
        )
        spinner_layout.add_widget(spinner_label)
        spinner_layout.add_widget(spinner)

        def on_ok_button_press(instance):
            popup.dismiss()
            selected_landmarks = [landmark for landmark, checkbox in self.selected_landmarks.items() if checkbox.active]
            self.nPca = int(spinner.text)  # Store the selected number
            print(f"Selected landmarks: {selected_landmarks}")
            print(f"Number of Dof of the task: {self.nPca}")
            self.extract_single_joint(selected_landmarks)  # Pass the selected landmarks to the function
        
        ok_button.bind(on_press=on_ok_button_press)

        # Wrap the scroll_view in a BoxLayout
        main_layout = BoxLayout(orientation='vertical')
        main_layout.add_widget(scroll_view)
        main_layout.add_widget(spinner_layout)

        # Add the OK button to the main layout
        main_layout.add_widget(ok_button)

        # Set the main layout as the content of the popup
        popup = Popup(title="Select Landmarks for Angle Extraction", content=main_layout, size_hint=(0.8, 0.8))
        popup.open()
        pass
    
    def open_image_popup(self, image_name):
        
        # Create a popup to display the image
        image_popup = Popup(title="Image", size_hint=(0.9, 0.9))
        image_layout = BoxLayout(orientation='vertical')
        image_widget = Image(source=image_name)
        image_layout.add_widget(image_widget)
        image_popup.content = image_layout
        image_popup.open()
    
    def extract_single_joint(self, joints_to_extract):
        
        print(joints_to_extract)
        # Match joints_to_extract to their index in left_landmarks and right_landmarks
        joint_indices_left = []
        joint_indices_right = []
        for joint in joints_to_extract:
            if joint in left_landmarks:
                joint_indices_left.append(left_landmarks.index(joint))
            elif joint in right_landmarks:
                joint_indices_right.append(right_landmarks.index(joint))

        # Select triplets corresponding to the indices in joint_indices_left and joint_indices_right
        selected_markers_left = [markers_left[i] for i in joint_indices_left]
        selected_markers_right = [markers_right[i] for i in joint_indices_right]

        # Combine the selected markers
        joints = selected_markers_left + selected_markers_right

        self.reference_joints_angles, self.reference_joint_file = self.extract_joints_from_file(self.reference_file, joints, joints_to_extract)
        self.comparison_joints_angles, self.comparison_joint_file = self.extract_joints_from_file(self.comparison_file, joints, joints_to_extract)
        
        cm_ref = cm.CoordinationMetric2(
                                list_files_angles=[self.reference_joint_file],
                                n_dof=len(self.reference_joints_angles.columns)-1,
                                list_angles = joints_to_extract, 
                                name='Reference', deg=True)
                            
        cm_comp = cm.CoordinationMetric2(
                                list_files_angles=[self.comparison_joint_file],
                                n_dof=len(self.comparison_joints_angles.columns)-1,
                                list_angles = joints_to_extract, 
                                name='Comparison', deg=True)

        # Clear the current layout
        self.clear_widgets()
       

        # Create a new layout for displaying the plots
        plot_layout = BoxLayout(orientation='vertical', size_hint_y=None)
        plot_layout.bind(minimum_height=plot_layout.setter('height'))
        scroll_view = ScrollView(size_hint=(1, 1))
        scroll_view.add_widget(plot_layout)

        # Create the first plot for reference joints angles
        plt.figure(figsize=(10, 5))
        for column in self.reference_joints_angles.columns[:-1]:  # Exclude the 'time' column
            plt.plot(self.reference_joints_angles['time'], self.reference_joints_angles[column], label=f"Reference: {column}")
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.ylim(0, 180)  # Set y-axis limits to 0-180 degrees
        plt.title('Reference Joint Angles')
        plt.legend()
        plt.tight_layout()
        plt.savefig("reference_plot.png")
        plt.close()

        # Create the second plot for comparison joints angles
        plt.figure(figsize=(10, 5))
        for column in self.comparison_joints_angles.columns[:-1]:  # Exclude the 'time' column
            plt.plot(self.comparison_joints_angles['time'], self.comparison_joints_angles[column], label=f"Comparison: {column}")
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (degrees)')
        plt.ylim(0, 180)  # Set y-axis limits to 0-180 degrees
        plt.title('Comparison Joint Angles')
        plt.legend()
        plt.tight_layout()
        plt.savefig("comparison_plot.png")
        plt.close()
        

        # Add the plots and corresponding videos to the new layout
        reference_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=400)
        r_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=450)

        reference_image = Image(source="reference_plot.png")
        reference_image.bind(on_touch_down=lambda instance, touch: self.open_image_popup("reference_plot.png") if instance.collide_point(*touch.pos) else None)
        reference_video = VideoPlayer(source=self.reference_file, state='play', options={'eos': 'loop'}, size_hint=(0.5, 1), volume=0)
        reference_label = Label(
            text="Reference Video / Video 1",
            color=(0, 1, 0, 1),  # Green color
            size_hint=(1, None),
            height=30,
            halign='center',
            valign='middle'
        )
        r_layout.add_widget(reference_label)
        reference_layout.add_widget(reference_image)
        reference_layout.add_widget(reference_video)   
        r_layout.add_widget(reference_layout) 
        with reference_layout.canvas.before:
            Color(0, 1, 0, 1)  # Green color
            self.rect = Line(rectangle=(reference_layout.x, reference_layout.y, reference_layout.width, reference_layout.height))
        reference_layout.bind(pos=lambda instance, value: setattr(self.rect, 'rectangle', (instance.x, instance.y, instance.width, instance.height)))
        reference_layout.bind(size=lambda instance, value: setattr(self.rect, 'rectangle', (instance.x, instance.y, instance.width, instance.height)))
       
        comparison_layout = BoxLayout(orientation='horizontal', size_hint_y=None, height=400)
        c_layout = BoxLayout(orientation='vertical', size_hint_y=None, height=450)
        comparison_image = Image(source="comparison_plot.png")
        comparison_image.bind(on_touch_down=lambda instance, touch: self.open_image_popup("comparison_plot.png") if instance.collide_point(*touch.pos) else None)
        if os.path.exists(self.comparison_file):
            comparison_video = VideoPlayer(source=self.comparison_file, state='play', options={'eos': 'loop'}, size_hint=(0.5, 1), volume=0)
            comparison_label = Label(
                text="Comparison Video / Video 2",
                color=(0, 0, 1, 1),  # Blue color
                size_hint=(1, None),
                height=30,
                halign='center',
                valign='middle'
            )
            c_layout.add_widget(comparison_label, index=0)
            comparison_layout.add_widget(comparison_image)
            comparison_layout.add_widget(comparison_video)
            c_layout.add_widget(comparison_layout)
            with comparison_layout.canvas.before:
                Color(0, 0, 1, 1)  # Blue color
                self.rect_comparison = Line(rectangle=(comparison_layout.x, comparison_layout.y, comparison_layout.width, comparison_layout.height))
            comparison_layout.bind(pos=lambda instance, _: setattr(self.rect_comparison, 'rectangle', (instance.x, instance.y, instance.width, instance.height)))
            comparison_layout.bind(size=lambda instance, _: setattr(self.rect_comparison, 'rectangle', (instance.x, instance.y, instance.width, instance.height)))
           
            
        else:
            print(f"Error: Video file not found at {self.comparison_file}")
            error_label = Label(text="Error: Video file not found.", color=(1, 0, 0, 1))
            comparison_layout.add_widget(error_label)

        plot_layout.add_widget(r_layout)
        plot_layout.add_widget(c_layout)

        # Add the new layout to the main view
        self.add_widget(scroll_view)
        
        # Run JcvPCA
        fig_jcvpca, res_jcvpca = cm_ref.plot_jcvpca(cm_comp, self.nPca)  
                   

               
        # Call the function to compute JsvCRP
        fig1_jsvcrp, fig2_jsvcrp, res_jsvcrp = cm_ref.plot_jsvcrp(cm_comp)
        
        print(res_jcvpca, res_jsvcrp)
        
        # Add a label to display the results
        results_label = Label(
            text="Results:",
            size_hint=(1, None),
            height=50,
            halign='center',
            valign='middle',
            color=(0, 0, 0, 1)
        )
        plot_layout.add_widget(results_label)
        
        summary = self.generate_interjoint_summary(res_jcvpca, res_jsvcrp, joints_to_extract)
        
        # Add the summary below the plots
        summary_label_interjoint = Label(
            text=summary,
            size_hint=(1, None),
            halign='center',
            valign='middle',
            markup=True,
            text_size=(Window.width * 0.8, None),
            color=(0, 0, 0, 1)
        )
        summary_label_interjoint.bind(
            texture_size=lambda instance, value: setattr(instance, 'height', value[1])
        )
    
        plot_layout.add_widget(summary_label_interjoint)
        
        # Add a round button with a "+" symbol
        round_button = Button(
            text="+",
            size_hint=(None, None),
            size=(60, 60),
            background_normal="",
            background_color=(0, 0.5, 1, 1),
            color=(1, 1, 1, 1),
            font_size='24sp',
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        round_button.bind(on_press=lambda instance: show_results([fig_jcvpca, fig2_jsvcrp]))
        plot_layout.add_widget(round_button)
        
        # Add a small space between the two buttons
        spacer = Label(size_hint=(1, None), height=10)
        plot_layout.add_widget(spacer)
        
        # Add a button with "More Metrics" text
        more_metrics_button = Button(
            text="More Metrics",
            size_hint=(1, None),
            height=50,
            background_color=(0, 0.5, 1, 1),
            color=(1, 1, 1, 1),
            font_size='18sp'
        )
        more_metrics_button.bind(on_press=lambda instance: self.more_metrics_popup(instance, cm_ref=cm_ref, cm_comp=cm_comp))
        plot_layout.add_widget(more_metrics_button)
        
    def more_metrics_popup(self, instance, cm_ref, cm_comp):
        
        # Create a popup with buttons for each metric
        popup_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # List of metrics
        metrics = [
            "Angle-Angle Plots",
            "Continuous Relative Phase",
            "Principal Component Analysis",
            "Statistical Correlation"
        ]

        # Add a button for each metric
        for metric in metrics:
            button = Button(text=metric, size_hint=(1, None), height=50)
            button.bind(on_press=lambda instance, m=metric: self.print_metric(m, cm_ref, cm_comp))
            popup_layout.add_widget(button)

        # Create and open the popup
        popup = Popup(title="Select a Metric", content=popup_layout, size_hint=(0.8, 0.8))
        popup.open()
    
    def print_metric(self, metric, cm_ref, cm_comp):
        
        if metric == 'Angle-Angle Plots':
            print("Angle-Angle Plots selected")
            res1 = cm_ref.plot_angle_angle()
            res2 = cm_comp.plot_angle_angle()
            # summary = generate_summary("Angle-Angle Plots", res1, res2)
            show_results([res1, res2])
        elif metric == 'Continuous Relative Phase':
            # Call the function to compute Continuous Relative Phase
            res1 = cm_ref.plot_continuous_relative_phase()
            res2 = cm_comp.plot_continuous_relative_phase()
            # summary = generate_summary("Continuous Relative Phase", res1, res2)
            show_results([res1, res2])
        elif metric == 'Principal Component Analysis':
            # Call the function to compute Principal Component Analysis
            res1 = cm_ref.plot_pca()
            res2 = cm_comp.plot_pca()
            # summary = generate_summary("Principal Component Analysis", res1, res2)
            show_results([res1, res2])
        elif metric == 'Statistical Correlation':
            # Call the function to compute Statistical Correlation
            res1 = cm_ref.plot_statistical_correlation()
            res2 = cm_comp.plot_statistical_correlation()
            # summary = generate_summary("Statistical Correlation", res1, res2)
            show_results([res1, res2])


      
        
    def generate_interjoint_summary(self, res_jcvpca, res_jsvcrp, joints_to_extract):
        
        summary = '[b]Joint Contribution :[/b] \n'
        
        contribution = False
        # Generate a summary for res_jcvpca
        for i, value in enumerate(res_jcvpca[0]):  # Iterate over the first line of res_jcvpca
            if value > 0.13:  # Check if the absolute value is greater than 0.15
                summary += f"Joint {joints_to_extract[i]} is more used in [color=00FF00]Video 2[/color] than in [color=0000FF]Video 1[/color]\n"
                contribution= True
        for i, value in enumerate(res_jcvpca[0]):  # Iterate over the first line of res_jcvpca
            if value < -0.13:  # Check if the absolute value is greater than 0.15
                summary += f"Joint {joints_to_extract[i]} is less used in [color=00FF00]Video 2[/color] than in [color=0000FF]Video 1[/color]\n"
                contribution= True
      
            
        if not contribution:   
            summary += "No significant differences were found in the joint contributions between the reference and comparison videos.\n\n"
        
        summary += '[b]Joint Synchronization :[/b] \n'
        for item, value in res_jsvcrp.items():
            print(value.iloc[0])
            if value.iloc[0] > 5000:
                joints = item.split('_')[-2:]  # Extract the two joints from the item name
                print(joints)
                summary += f"The joints {joints[0]} and {joints[1]} are synchronized differently in both videos.\n"
        
        print(summary)
        if summary == '':
            summary = "No significant differences were found between the reference and comparison videos."
       
        return summary
        
    def extract_joints_from_file(self, file, joints, joints_names):
        
        output_csv = os.path.join(os.path.dirname(file), file.split('/')[-1].split('.')[0] + '.csv')
        data = pd.read_csv(output_csv)
        df_results = pd.DataFrame(columns=joints_names)

        for i in data['frame_number'].unique():
            #if i%10 == 0:
                #Extracting angles only in the right side of participants
                s = pd.Series({joint: 0 for joint in joints_names})
                s.name=i
                df_results = pd.concat([df_results, s.to_frame().T])
                print(df_results)
                for k,j in enumerate(joints_names) :
                    A = data[(data['frame_number']==i) & (data['landmark']==joints[k][0])].head(1).values
                    B = data[(data['frame_number']==i) & (data['landmark']==joints[k][1])].head(1).values
                    C = data[(data['frame_number']==i) & (data['landmark']==joints[k][2])].head(1).values
            
                    _, angle = angle_between_points(A[:,2:5].reshape(3), B[:,2:5].reshape(3), C[:,2:5].reshape(3))
                    df_results.loc[i][j]=angle
                    
        for j in joints_names : 
            df_results[j] = low_pass_filter(order, normalized_cutoff,  df_results[j].to_numpy(dtype=np.float64))
                
        df_results = df_results.reset_index()
        cols = df_results.columns
        cols = list(cols[1:]) + [cols[0]]
        df_results = df_results[cols]
        df_results['index'] = df_results['index']/24
        df_results['time'] = df_results['index']
        df_results = df_results.drop(columns=['index'])
        output_csv_no_ext = os.path.splitext(output_csv)[0]
        df_results.to_csv(output_csv_no_ext + '_angle.csv', index=False)
        
        # Compute the velocity of each joint
        for j in joints_names:
            df_results[j + '_velocity'] = df_results[j].diff() * 24  # Multiply by 24 to account for the sampling rate
        
        # Compute the maximum velocity for each joint
        max_velocities = {j: df_results[j + '_velocity'].max() for j in joints_names}
        print(f"Max velocities: {max_velocities}")
        # Find the joint with the highest maximum velocity
        joint_with_highest_velocity = max(max_velocities, key=max_velocities.get)
        print(f"Joint with the highest maximum velocity: {joint_with_highest_velocity}")

        # Compute 5% of the maximum velocity for each joint
        threshold_velocities = 0.2 * max_velocities[joint_with_highest_velocity] 

        # Find the first index where the joint_with_highest_velocity exceeds the threshold
        start_indices = {}
        movement_start_index = df_results[df_results[joint_with_highest_velocity + '_velocity'] > threshold_velocities].index.min()

        
        print(f"Movement starts at index: {movement_start_index}")

        # Crop df_results to keep only rows with index greater than or equal to movement_start_index
        df_results = df_results.iloc[movement_start_index:].reset_index(drop=True)
        # Drop velocity columns
        velocity_columns = [col for col in df_results.columns if '_velocity' in col]
        df_results = df_results.drop(columns=velocity_columns)
        
        return df_results, output_csv_no_ext + '_angle.csv'

   
        


class InterJointApp(App):
    def build(self):
        request_permissions([
            Permission.READ_EXTERNAL_STORAGE,
            Permission.WRITE_EXTERNAL_STORAGE,
        ])
        root = BoxLayout(orientation='vertical')
        header = Header()
        root.add_widget(header)
        root.add_widget(HomeView())
        return root
   
    
    
InterJointApp = InterJointApp()
InterJointApp.run()
