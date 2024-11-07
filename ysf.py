from funasr import AutoModel
import torch_npu
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "iic/SenseVoiceSmall"
torch_npu.npu.set_device("npu:0")

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",  
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="npu:0",
)
#model.to("npu:3")

# en
res = model.generate(
    input="./ysf.wav",
    cache={},
    language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print(text)