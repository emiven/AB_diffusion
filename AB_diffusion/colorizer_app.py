
from ipyevents import Event
import io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import ipywidgets as widgets
from IPython.display import display
from AB_diffusion.color_handling import normalize_lab, de_normalize_lab, PointColorConversions
import kornia.color as kacolor



class ModelWrapper:
    """
    Wrapper class for the ABUnet model.
    """
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    # Colorizes the greyscaøe image when user clics the colorize button
    def colorize(self, input_image_L_tensor, user_hints_tensor, output_count):
        conditioning = torch.cat([input_image_L_tensor, user_hints_tensor], dim=1)
        conditioning = conditioning.repeat(output_count, 1, 1, 1)
        conditioning = normalize_lab(conditioning)
        output_AB = self.model.sample(conditioning.to(self.device))

        self.colorization_LAB_tensor = de_normalize_lab(torch.cat([conditioning[:, :1, :, :], output_AB.to("cpu")], dim=1))

# Handles the image processing and conversions   
class ImageProcessor:
    """
    A class for handling image operations for the colorizer app.
    """
    def __init__(self, image):
        self.image = image.copy()
        self.input_image_LAB_tensor = self.convert_to_lab_tensor(self.image.copy())
        self.input_image_L_tensor = self.input_image_LAB_tensor[:, :1, :, :]
        self.user_hints_tensor = torch.zeros_like(self.input_image_LAB_tensor[:, 1:, :, :])
        
        self.point_color_conversions = PointColorConversions()

    def convert_to_lab_tensor(self, image):
        return kacolor.rgb_to_lab(transforms.ToTensor()(image).unsqueeze(0))

    def lab_tensor_to_image(self, lab_tensor):
        output_image = transforms.ToPILImage()(kacolor.lab_to_rgb (lab_tensor.squeeze(0).detach().cpu()))
        return output_image

    def apply_hints_to_image(self, x, y, color, sampling_size):
        
        color_lab = self.point_color_conversions.hex_to_lab(color)
    
        half_sampling = sampling_size // 2

        x_start = max(0, x - half_sampling)
        y_start = max(0, y - half_sampling)
        x_end = min(self.image.width, x + half_sampling)
        y_end = min(self.image.height, y + half_sampling)
    
        # Apply color to image tensor
        self.user_hints_tensor[:,0,y_start:y_end,x_start:x_end] = color_lab[1]
        self.user_hints_tensor[:,1,y_start:y_end,x_start:x_end] = color_lab[2]


    def clear_inputs(self):
        self.user_hints_tensor = torch.zeros_like(self.user_hints_tensor)

    def get_input_tensors(self):
        return self.input_image_L_tensor, self.user_hints_tensor

    def get_input_image(self):
        return self.image

    def get_hinted_image(self):
        temp_lab = torch.cat([self.input_image_L_tensor, self.user_hints_tensor], dim=1)
        return self.lab_tensor_to_image(temp_lab)

class WidgetManager:
    """
    A class for managing layout, widgets and their functionality.
    """
    def __init__(self, image_processor, model_wrapper):
        self.image_processor = image_processor
        self.model_wrapper = model_wrapper
        
        self.colorized_images_bytes = None

        self.image_widget = self.create_image_widget()
        self.main_colorized_image_widget = self.create_main_colorized_image_widget()
        self.colorized_image_grid = self.create_colorized_image_grid()
        self.color_picker = self.create_color_picker()
        self.hint_size_slider = self.create_hint_size_slider()
        self.output_count_slider = self.create_output_count_slider()
        self.colorize_button = self.create_colorize_button()
        self.clear_button = self.create_clear_button()
        self.export_button = self.create_export_button()
    
    

    def display(self):

        # Create labels
        input_image_label = widgets.Label('Input Image')
        colorization_label = widgets.Label('Colorization')

        input_image_box = widgets.VBox([input_image_label, self.image_widget])
        colorized_image_grid_box = widgets.VBox([widgets.Label('Click to select'), self.colorized_image_grid])
    

        colorization_box = widgets.HBox([widgets.VBox([colorization_label, self.main_colorized_image_widget]), colorized_image_grid_box], box_style='info')

        image_layout = widgets.HBox([
            input_image_box,
            colorization_box, 
        ])

        # Create a layout for the controls
        control_layout = widgets.VBox([
            self.color_picker, 
            self.hint_size_slider, 
            self.output_count_slider, 
            self.colorize_button,
            self.clear_button,
            self.export_button,
        ])

        layout = widgets.VBox([image_layout,control_layout])
        display(layout)

    # Creates the additional colorized image grid    
    def create_colorized_image_grid(self):
        colorized_image_grid = widgets.GridBox([], layout=widgets.Layout(grid_template_columns="repeat(3, 100px)"),align_items='center')
        return colorized_image_grid
    
    # "Main" colorized image widget
    def create_main_colorized_image_widget(self):
        main_colorized_image_widget = widgets.Image(format='png',description='Colorized Image')
        main_colorized_image_widget.layout = widgets.Layout(object_fit='contain', height='auto', width='300px')
        return main_colorized_image_widget
    # Wigdet for the loaded input image
    def create_image_widget(self):
        
        image_widget = widgets.Image(format='png',description='Input Image')

        image_widget.layout = widgets.Layout(object_fit='contain', height='auto', width='300px')
        # Set the initial image data
        image_widget.value = self.to_bytes(self.image_processor.get_input_image())

        event = Event(source=image_widget, watched_events=['click'])
        event.on_dom_event(self.on_image_click)

        return image_widget
    
    def create_color_picker(self):
        color_picker = widgets.ColorPicker(value='#000000', description='Color:')
        return color_picker
    def create_hint_size_slider(self):
        hint_size_slider = widgets.IntSlider(value=4, min=2, max=10, description='Hint Size:')
        return hint_size_slider
    def create_output_count_slider(self):
        output_count_slider = widgets.IntSlider(value=1, min=1, max=10, description='Output Count:')
        return output_count_slider
    
    def create_colorize_button(self):
        colorize_button = widgets.Button(description='Colorize!')
        colorize_button.on_click(self.on_colorize_button_click)
        return colorize_button
    def create_clear_button(self):
        clear_button = widgets.Button(description='Clear')
        clear_button.on_click(self.on_clear_button_click)
        return clear_button
    
    def create_export_button(self):
        self.export_filename_text = widgets.Text(value='output.png', description='Filename:')
        export_button = widgets.Button(description='Export')
        export_button.on_click(self.on_export_button_click)
        return widgets.VBox([self.export_filename_text, export_button])

    def on_export_button_click(self, _):
        # Get the image data from the widget
        image_data = self.main_colorized_image_widget.value

        if not image_data:
            print("No image data to export.")
            return

        # Convert the image data to a PIL Image
        try:
            image = Image.open(io.BytesIO(image_data))
        except IOError:
            print("Cannot convert image data to PIL Image. The data may not be in a valid image format.")
            return

        # Save the image with the filename from the text widget
        filename = self.export_filename_text.value
        try:
            image.save(filename)
            print(f"Image exported as {filename}.")
        except IOError:
            print(f"Cannot save image as {filename}.")
    
    # Additional image wigdets for multiple colorizations, attached with an click event for selection
    def create_colorized_image_widget(self, image_bytes):
        img_widget = widgets.Image(value=image_bytes, format='png')
        img_widget.layout = widgets.Layout(object_fit='contain', height='auto', width='100px')
    
        def on_click(event):
            self.on_additional_colorization_click(image_bytes)

        event_handler = Event(source=img_widget, watched_events=['click'])
        event_handler.on_dom_event(on_click)

        return img_widget

    def on_clear_button_click(self, _):
        self.image_processor.clear_inputs()
        self.image_widget.value = self.to_bytes(self.image_processor.get_hinted_image())
    

    # Handles the colorization process, updates the widgets with the colorized images
    def on_colorize_button_click(self, _):
        input_image_L_tensor, user_hints_tensor = self.image_processor.get_input_tensors()
        output_count = self.output_count_slider.value
    
        self.model_wrapper.colorize(input_image_L_tensor, user_hints_tensor, output_count)
    
        model_output_LAB_tensor = self.model_wrapper.colorization_LAB_tensor
        # Convert the colorized images from tensors ->to bytes
        self.colorized_images_bytes = [self.to_bytes(self.image_processor.lab_tensor_to_image(img)) for img in model_output_LAB_tensor]
    
        # Main colorized image widget gets the first colorized image
        self.main_colorized_image_widget.value = self.colorized_images_bytes[0]
    
        # Widgets for additional colorizations
        additional_colorizations_widgets = []

        if output_count > 1:
            for img_bytes in self.colorized_images_bytes:
                img_widget = self.create_colorized_image_widget(img_bytes)
                additional_colorizations_widgets.append(img_widget)
    
        # Update the grid
        self.colorized_image_grid.children = additional_colorizations_widgets
    
    # Updates the main colorized image widget with the clicked image
    def on_additional_colorization_click(self, image_bytes):
        
        self.main_colorized_image_widget.value = image_bytes
    
    # Handles the click event on the input image widget, and application of user hints to the image
    def on_image_click(self, event):
        x = event['dataX']
        y = event['dataY']
        color = self.color_picker.value
        sampling_size = self.hint_size_slider.value

        self.image_processor.apply_hints_to_image(x, y, color, sampling_size)

        hinted_image = self.image_processor.get_hinted_image()
        self.image_widget.value = self.to_bytes(hinted_image)
    
    def to_bytes(self, pil_image):
        byte_arr = io.BytesIO()
        pil_image.save(byte_arr, format='PNG')
        return byte_arr.getvalue()

class ColorizerApp:
    """
    A class for running the colorizer app.
    """ 
    def __init__(self, image, model, device="cpu"):
        self.image_processor = ImageProcessor(image)
        self.model_wrapper = ModelWrapper(model, device)
        self.widget_manager = WidgetManager(self.image_processor, self.model_wrapper)

    def run(self):
        self.widget_manager.display()


