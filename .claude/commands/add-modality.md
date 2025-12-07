# Add AI Modality Support

You are a machine learning specialist helping to add support for new AI modalities to NeuralTrainer-NET.

## Your Task

Add support for a new AI modality (audio, speech, vision variant, etc.). The user will specify: $ARGUMENTS

## Supported Modalities Roadmap

- **Text Generation** - Implemented
- **Image Classification** - Implemented
- **Audio Transcription** - Planned
- **Audio Generation** - Planned
- **Text-to-Speech** - Planned
- **Speech-to-Speech** - Planned
- **Image Generation** - Planned
- **Video Processing** - Planned

## Steps to Follow

1. **Research the Modality**
   - Understand input data format and preprocessing needs
   - Identify standard model architectures used
   - Determine required layers (existing or new)
   - Research common datasets for testing

2. **Create Preprocessing Pipeline**
   - Add directory: `NeuralNetwork/Processing/{Modality}/`
   - Implement data loading and normalization
   - Handle format conversions (e.g., audio to spectrograms)
   - Create tokenization/encoding if needed

3. **Add Required Layers**
   - Identify missing layer types
   - Implement using `/add-layer` command patterns
   - Consider CUDA acceleration for compute-heavy layers

4. **Create Inference Utilities**
   - Add to `NeuralNetwork/Inference/`
   - Implement output decoding/generation
   - Handle streaming output if applicable

5. **Add Training Example**
   - Create method in `NET_Keras/Program.cs`
   - Include data loading, model definition, training loop
   - Add inference demonstration

## Directory Structure for New Modality

```
NeuralNetwork/
├── Processing/
│   └── {Modality}/
│       ├── {Modality}PreProcessing.cs    # Data preprocessing
│       └── {Modality}DataLoader.cs       # Dataset loading
├── Inference/
│   └── {Modality}Generator.cs            # Output generation
├── Layers/
│   └── {Modality-specific layers}        # If needed
```

## Audio Modality Template

```csharp
// NeuralNetwork/Processing/Audio/AudioPreProcessing.cs
namespace NeuralNetwork.Processing.Audio
{
    public class AudioPreProcessing
    {
        public int SampleRate { get; set; } = 16000;
        public int FFTSize { get; set; } = 512;
        public int HopLength { get; set; } = 160;
        public int MelBins { get; set; } = 80;

        // Convert audio waveform to mel spectrogram
        public float[,] WaveformToMelSpectrogram(float[] waveform)
        {
            // STFT -> Mel filterbank -> Log scale
        }

        // Load audio file and resample
        public float[] LoadAudio(string path)
        {
            // Load WAV/MP3 and resample to target rate
        }

        // Normalize audio
        public float[] Normalize(float[] audio)
        {
            // Peak or RMS normalization
        }
    }
}
```

## Common Architectures by Modality

### Audio Transcription (ASR)
- Input: Mel spectrograms
- Layers: Conv2D -> LSTM/Transformer -> CTC decoder
- Output: Text tokens

### Text-to-Speech (TTS)
- Input: Text tokens + speaker embedding
- Layers: Encoder -> Attention -> Decoder -> Vocoder
- Output: Audio waveform

### Audio Generation
- Input: Conditioning (text/audio)
- Layers: Diffusion/GAN/Autoregressive
- Output: Audio waveform

## Related Agents

- Use `/data-pipeline` to create robust data loading
- Use `/add-layer` for modality-specific layers
- Use `/add-cuda` to GPU-accelerate heavy preprocessing
- Use `/test-model` to validate end-to-end pipeline
- Use `/benchmark` to measure training throughput
- Use `/architecture` to document the modality design
- Use `/coordinate` for complex multi-step implementations

## Quality Checklist

- [ ] Preprocessing handles common formats
- [ ] Data normalization implemented
- [ ] Model architecture appropriate for modality
- [ ] Inference produces expected output type
- [ ] Training example works end-to-end
- [ ] Documentation updated in CLAUDE.md
- [ ] Data pipeline efficient (use `/benchmark`)
- [ ] End-to-end test passing (use `/test-model`)
