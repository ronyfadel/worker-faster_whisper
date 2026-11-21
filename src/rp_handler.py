"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import os
import tempfile

from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
from runpod.version import __version__ as runpod_version
import predict
from audio_download import download_audio
from cuda_oom import is_cuda_out_of_memory, out_of_memory_result


MODEL = predict.Predictor()
MODEL.setup()


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction
    '''
    job_input = job['input']

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    if job_input.get('audio', False):
        with rp_debugger.LineTimer('download_step'):
            # Raises AudioDownloadError ("Couldn't download the audio …") when it can't, so the
            # job fails with a clear reason instead of "No such file or directory: 'None'".
            audio_input = download_audio(job_input['audio'])

    if job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])

    try:
        with rp_debugger.LineTimer('prediction_step'):
            # Every validated input other than the audio sources and the model name
            # maps 1:1 onto a Predictor.predict keyword argument.
            predict_kwargs = {
                key: value
                for key, value in job_input.items()
                if key not in ("audio", "audio_base64", "model")
            }
            whisper_results = MODEL.predict(
                audio=audio_input,
                model_name=job_input["model"],
                **predict_kwargs,
            )
    except RuntimeError as error:  # torch.cuda.OutOfMemoryError is a RuntimeError too
        if not is_cuda_out_of_memory(error):
            raise
        # memory left behind by earlier jobs: fail this one and have RunPod replace the worker
        return out_of_memory_result(error, runpod_version)
    finally:
        with rp_debugger.LineTimer('cleanup_step'):
            rp_cleanup.clean(['input_objects'])
            # The downloaded or decoded audio lives in a temp file; long recordings are 100 MB+.
            try:
                os.remove(audio_input)
            except OSError:
                pass

    return whisper_results


runpod.serverless.start({"handler": run_whisper_job})
