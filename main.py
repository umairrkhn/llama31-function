import functions_framework
from flask import request, jsonify
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

def convert_protobuf_to_dict(protobuf_obj):
    """Convert a Protobuf object to a dictionary."""
    return json_format.MessageToDict(protobuf_obj, preserving_proto_field_name=True)

def predict_custom_trained_model_sample(
    project: str,
    endpoint_id: str,
    instances: dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    """Make a prediction using a custom-trained model."""
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    
    # Convert instances to the format required by the model
    instances = [json_format.ParseDict(instances, Value())]
    parameters = json_format.ParseDict({}, Value())
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    
    try:
        response = client.predict(endpoint=endpoint, instances=instances, parameters=parameters)
        # Convert predictions to a serializable format
        predictions = []
        for prediction in response.predictions:
            # Check if the prediction is a Protobuf message or already in a serializable format
            if isinstance(prediction, bytes):
                # Handle byte array case if needed (e.g., if predictions are in bytes format)
                predictions.append(prediction.decode('utf-8'))
            elif hasattr(prediction, 'DESCRIPTOR'):
                # Convert Protobuf object to dict
                predictions.append(convert_protobuf_to_dict(prediction))
            else:
                # Directly append if already in JSON format
                predictions.append(prediction)
        return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        raise

@functions_framework.http
def predict(request):
    """HTTP Cloud Function to predict using a custom-trained model."""
    try:
        data = request.get_json()
        if not data or "instances" not in data:
            return jsonify({"error": "Missing 'instances' in request body"}), 400
        
        project = "17113908719"
        endpoint_id = "5768264498708217856"
        instances = data.get("instances")

        # Make prediction request
        predictions = predict_custom_trained_model_sample(
            project=project,
            endpoint_id=endpoint_id,
            instances=instances,
        )
        
        # Return predictions
        return jsonify({"predictions": predictions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
