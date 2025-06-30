from flask import Flask, request, jsonify
import logging
from logging.handlers import RotatingFileHandler
import traceback
import os
import re

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
handler = RotatingFileHandler('stabilizer_api.log', maxBytes=10485760, backupCount=5)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
app.logger.addHandler(handler)

CONFIG_PATH = "config.yaml"
CONFIG_BACKUP = "config.yaml.bak"

def backup_config():
    """Create a backup of the current config file"""
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_BACKUP, 'w') as backup_file:
                with open(CONFIG_PATH, 'r') as original_file:
                    backup_file.write(original_file.read())
            return True
        return False
    except Exception as e:
        app.logger.error(f"Error creating backup: {str(e)}")
        return False

def update_config_from_json(json_data):
    """Update specific parameters in the config file based on JSON data"""
    try:
        # Don't update video_source
        if 'videoSource' in json_data:
            app.logger.info("Ignoring videoSource parameter as requested")
            del json_data['videoSource']
        
        # Load the config file as text
        with open(CONFIG_PATH, 'r') as file:
            config_text = file.read()
        
        # Create backup
        backup_config()
        
        # Track what we've updated
        updated_params = []
        
        # Parameter mappings
        mappings = {
            # Mode section
            "width": ("mode:", "width:", "int"),
            "height": ("mode:", "height:", "int"),
            "optimizeFps": ("mode:", "optimize_fps:", "int_bool"),
            "useCuda": ("mode:", "use_cuda:", "int_bool"),
            "enhancerEnabled": ("mode:", "enhancer_enabled:", "int_bool"),
            "rollCorrectionEnabled": ("mode:", "roll_correction_enabled:", "int_bool"),
            "stabilizerEnabled": ("mode:", "stabilizer_enabled:", "int_bool"),
            
            # Camera section
            "threadedQueueMode": ("camera:", "threaded_queue_mode:", "bool"),
            "logging": ("camera:", "logging:", "bool"),
            "timeDelay": ("camera:", "time_delay:", "int"),
            "threadTimeout": ("camera:", "thread_timeout:", "int"),
            
            # Enhancer section
            "brightness": ("enhancer:", "brightness:", "float"),
            "contrast": ("enhancer:", "contrast:", "float"),
            "gamma": ("enhancer:", "gamma:", "float"),
            "enableWhiteBalance": ("enhancer:", "enable_white_balance:", "int_bool"),
            "wbStrength": ("enhancer:", "wb_strength:", "float"),
            "enableVibrance": ("enhancer:", "enable_vibrance:", "int_bool"),
            "vibranceStrength": ("enhancer:", "vibrance_strength:", "float"),
            "enableUnsharp": ("enhancer:", "enable_unsharp:", "int_bool"),
            "sharpness": ("enhancer:", "sharpness:", "float"),
            "enableDenoise": ("enhancer:", "enable_denoise:", "int_bool"),
            "denoiseStrength": ("enhancer:", "denoise_strength:", "int"),
            "enableClahe": ("enhancer:", "enable_clahe:", "int_bool"),
            "claheClipLimit": ("enhancer:", "clahe_clip_limit:", "float"),
            "claheTileGridSize": ("enhancer:", "clahe_tile_grid_size:", "int"),
            
            # Roll correction section
            "rollScaleFactor": ("roll_correction:", "scale_factor:", "float"),
            "cannyThresholdLow": ("roll_correction:", "canny_threshold_low:", "int"),
            "cannyThresholdHigh": ("roll_correction:", "canny_threshold_high:", "int"),
            "cannyAperture": ("roll_correction:", "canny_aperture:", "int"),
            "houghRho": ("roll_correction:", "hough_rho:", "float"),
            "houghThreshold": ("roll_correction:", "hough_threshold:", "int"),
            "angleSmoothingAlpha": ("roll_correction:", "angle_smoothing_alpha:", "float"),
            "angleDecay": ("roll_correction:", "angle_decay:", "float"),
            "angleFilterMin": ("roll_correction:", "angle_filter_min:", "float"),
            "angleFilterMax": ("roll_correction:", "angle_filter_max:", "float"),
            
            # Stabilizer section
            "smoothingRadius": ("stabilizer:", "smoothing_radius:", "int"),
            "borderType": ("stabilizer:", "border_type:", "string"),
            "borderSize": ("stabilizer:", "border_size:", "int"),
            "cropAndZoom": ("stabilizer:", "crop_n_zoom:", "int_bool"),
            "strength": ("stabilizer:", "strength:", "int"),
        }
        
        # Process each parameter
        for param_name, value in json_data.items():
            if param_name in mappings:
                section, key, value_type = mappings[param_name]
                
                # Format the value according to its type
                if value_type == "int_bool":
                    formatted_value = "1" if value else "0"
                elif value_type == "bool":
                    formatted_value = "true" if value else "false"
                elif value_type == "string":
                    formatted_value = f'"{value}"'
                elif value_type == "int":
                    formatted_value = str(int(value))
                elif value_type == "float":
                    formatted_value = str(float(value))
                else:
                    formatted_value = str(value)
                
                # Create search pattern
                pattern = rf'{re.escape(section)}.*?{re.escape(key)}.*?(\d+|\d+\.\d+|"[^"]*"|true|false)'
                
                # Try to find and replace the value
                match = re.search(pattern, config_text, re.DOTALL)
                if match:
                    start_key_pos = config_text.find(key, match.start())
                    if start_key_pos != -1:
                        # Find the beginning of the value
                        value_start = start_key_pos + len(key)
                        while value_start < len(config_text) and config_text[value_start].isspace():
                            value_start += 1
                        
                        # Find the end of the value
                        value_end = value_start
                        while value_end < len(config_text) and config_text[value_end] not in ('#', '\n'):
                            value_end += 1
                        
                        # Extract the value
                        old_value = config_text[value_start:value_end].strip()
                        
                        # Replace the value
                        config_text = config_text[:value_start] + " " + formatted_value + config_text[value_end:]
                        
                        updated_params.append(f"{section.strip(':')} -> {key.strip(':')} = {formatted_value}")
                else:
                    app.logger.warning(f"Could not find parameter: {section} {key}")
        
        # Write the updated config
        with open(CONFIG_PATH, 'w') as file:
            file.write(config_text)
        
        app.logger.info(f"Updated {len(updated_params)} parameters: {', '.join(updated_params)}")
        return True
    
    except Exception as e:
        app.logger.error(f"Error updating config: {str(e)}\n{traceback.format_exc()}")
        return False

@app.route('/stabilization', methods=['POST'])
def update_stabilization():
    """API endpoint to receive stabilization parameters and update config"""
    try:
        json_data = request.get_json()
        
        if not json_data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400
        
        app.logger.info(f"Received parameters: {json_data}")
        
        # Update the configuration
        if update_config_from_json(json_data):
            return jsonify({
                "status": "success", 
                "message": "Configuration updated successfully"
            }), 200
        else:
            return jsonify({
                "status": "error", 
                "message": "Failed to update configuration"
            }), 500
        
    except Exception as e:
        error_details = traceback.format_exc()
        app.logger.error(f"Error processing request: {str(e)}\n{error_details}")
        return jsonify({
            "status": "error", 
            "message": f"Internal server error: {str(e)}"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    # Use 0.0.0.0 to accept connections from any IP
    app.run(host='192.168.144.101', port=5748, debug=False)
