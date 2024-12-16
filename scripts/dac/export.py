
import torch
import torch.nn as nn
import cached_conv as cc
import nn_tilde
import dac

class DummyRTTest(nn_tilde.Module):

    def __init__(self):
        super().__init__()
        self.register_method(
            "forward", 
            in_channels=1, 
            in_ratio=1,
            out_channels=1,
            out_ratio=1,  
        )

    def forward(self, x):
        print(f"block with shape: {x.shape}")
        return x

class ScriptedDAC(nn_tilde.Module):

    def __init__(self, 
        pretrained: dac.DAC, 
    ):
        super().__init__()
        self.pretrained = pretrained

        self.register_method(
            "encode", 
            in_channels=self.pretrained.num_channels, 
            in_ratio=1, 
            out_channels=self.pretrained.n_codebooks, 
            out_ratio=self.pretrained.hop_length, 
            input_labels=["(signal) model input 0"],
            output_labels=[f"(signal) model output {i}" for i in range(self.pretrained.n_codebooks)]
        )
        self.register_method(
            "decode", 
            in_channels=self.pretrained.n_codebooks, 
            in_ratio=self.pretrained.hop_length, 
            out_channels=self.pretrained.num_channels, 
            out_ratio=1, 
            input_labels=[f"(signal) model input {i}" for i in range(self.pretrained.n_codebooks)],
            output_labels=["(signal) model output 0"]
        )
        self.register_method(
            "forward", 
            in_channels=self.pretrained.num_channels, 
            in_ratio=1, 
            out_channels=self.pretrained.num_channels, 
            out_ratio=1, 
            input_labels=["(signal) model input 0"],
            output_labels=["(signal) model output 0"]
        )

    @torch.jit.export
    def encode(self, x):
        # breakpoint()
        return self.pretrained.encode(x)["codes"].float()

    @torch.jit.export
    def decode(self, z):
        z = self.pretrained.quantizer.from_codes(z.int())[0]
        return self.pretrained.decode(z)

    def forward(self, x):
        z = self.encode(x)
        y = self.decode(z)
        return y

    
cc.use_cached_conv(True)

# Load the pretrained model
# Download a model
model_path = "runs/dim=512/best/dac/weights.pth"
pretrained = dac.DAC.load(model_path)
pretrained.eval()
for m in pretrained.modules():
    if hasattr(m, "weight_g"):
        nn.utils.remove_weight_norm(m)

# Create a scripted model
model = ScriptedDAC(pretrained)

testmodel = DummyRTTest()

x = torch.zeros(1, pretrained.num_channels, 2**14)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
    z = model.encode(x)
    y = model.decode(z)

print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))


# Save the model
model_path = "dac_scripted.pt"
model.export_to_ts(model_path)

testmodel.export_to_ts("rt_test.pt")
