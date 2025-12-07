# Export Model

You are a model serialization specialist for NeuralTrainer-NET, helping to save, load, and export models.

## Your Task

Export or convert trained models. The user will specify what they need: $ARGUMENTS

## Export Formats

| Format | Use Case | Interoperability |
|--------|----------|------------------|
| Native (.nn) | NeuralTrainer-NET models | This framework only |
| ONNX (.onnx) | Cross-platform inference | PyTorch, TensorFlow, etc. |
| Weights (.bin) | Just the weights | Any framework |
| JSON Config | Model architecture | Human-readable |

## Steps to Follow

1. **Understand Requirements**
   - Which format needed?
   - Full model or weights only?
   - Target platform/framework?

2. **Prepare Model**
   - Ensure model is trained/loaded
   - Verify model structure
   - Get weight shapes

3. **Implement Export**
   - Serialize weights
   - Save architecture config
   - Write appropriate format

4. **Verify Export**
   - Load and run inference
   - Compare outputs
   - Test on target platform

## Native Format Implementation

The native format is already implemented. Review:
- `NeuralNetwork/Ext/SaveSequential.cs`
- `NeuralNetwork/Ext/LoadSequential.cs`

```csharp
// Save model
model.Save("model.nn");

// Load model
var model = Sequential.Load("model.nn");
```

## ONNX Export Implementation

```csharp
namespace NeuralNetwork.Export
{
    public class OnnxExporter
    {
        private Sequential model;
        private int opsetVersion = 13;

        public OnnxExporter(Sequential model)
        {
            this.model = model;
        }

        public void Export(string path, int[] inputShape)
        {
            using var file = File.Create(path);

            // Build ONNX graph
            var graph = BuildGraph(inputShape);

            // Create model
            var onnxModel = new OnnxModel
            {
                IrVersion = 8,
                OpsetImport = new[] { new OpsetImport { Version = opsetVersion } },
                Graph = graph
            };

            // Serialize (using protobuf)
            onnxModel.WriteTo(file);
        }

        private OnnxGraph BuildGraph(int[] inputShape)
        {
            var nodes = new List<OnnxNode>();
            var initializers = new List<OnnxTensor>();

            // Input
            var inputs = new List<OnnxValueInfo>
            {
                new OnnxValueInfo
                {
                    Name = "input",
                    Type = CreateTensorType(inputShape)
                }
            };

            string lastOutput = "input";

            // Convert each layer
            for (int i = 0; i < model.Layers.Count; i++)
            {
                var layer = model.Layers[i];
                string outputName = $"layer_{i}_output";

                var (node, tensors) = ConvertLayer(layer, lastOutput, outputName, i);
                nodes.Add(node);
                initializers.AddRange(tensors);

                lastOutput = outputName;
            }

            // Output
            var outputs = new List<OnnxValueInfo>
            {
                new OnnxValueInfo { Name = lastOutput }
            };

            return new OnnxGraph
            {
                Nodes = nodes,
                Initializers = initializers,
                Inputs = inputs,
                Outputs = outputs
            };
        }

        private (OnnxNode, List<OnnxTensor>) ConvertLayer(Layer layer, string input, string output, int idx)
        {
            return layer switch
            {
                Dense dense => ConvertDense(dense, input, output, idx),
                LSTM lstm => ConvertLSTM(lstm, input, output, idx),
                Conv2D conv => ConvertConv2D(conv, input, output, idx),
                Embedding emb => ConvertEmbedding(emb, input, output, idx),
                _ => throw new NotSupportedException($"Layer type {layer.GetType().Name} not supported for ONNX export")
            };
        }

        private (OnnxNode, List<OnnxTensor>) ConvertDense(Dense layer, string input, string output, int idx)
        {
            string weightsName = $"dense_{idx}_weights";
            string biasName = $"dense_{idx}_bias";
            string matmulOutput = $"dense_{idx}_matmul";

            var tensors = new List<OnnxTensor>
            {
                CreateTensor(weightsName, layer.Weights),
                CreateTensor(biasName, layer.Biases)
            };

            var nodes = new List<OnnxNode>
            {
                // MatMul
                new OnnxNode
                {
                    OpType = "MatMul",
                    Inputs = new[] { input, weightsName },
                    Outputs = new[] { matmulOutput }
                },
                // Add bias
                new OnnxNode
                {
                    OpType = "Add",
                    Inputs = new[] { matmulOutput, biasName },
                    Outputs = new[] { output }
                }
            };

            // Add activation if needed
            // ...

            return (nodes[0], tensors);
        }
    }
}
```

## Weights-Only Export

```csharp
public class WeightExporter
{
    public static void ExportWeights(Sequential model, string path)
    {
        using var writer = new BinaryWriter(File.Create(path));

        // Write header
        writer.Write("NNWGT");  // Magic number
        writer.Write(1);        // Version
        writer.Write(model.Layers.Count);

        foreach (var layer in model.Layers)
        {
            // Layer name
            writer.Write(layer.GetType().Name);

            if (layer.Weights != null)
            {
                // Weight shape
                writer.Write(layer.Weights.GetLength(0));
                writer.Write(layer.Weights.GetLength(1));

                // Weight data
                foreach (float val in layer.Weights)
                    writer.Write(val);
            }
            else
            {
                writer.Write(0);
                writer.Write(0);
            }

            if (layer.Biases != null)
            {
                writer.Write(layer.Biases.Length);
                foreach (float val in layer.Biases)
                    writer.Write(val);
            }
            else
            {
                writer.Write(0);
            }
        }
    }

    public static void ImportWeights(Sequential model, string path)
    {
        using var reader = new BinaryReader(File.OpenRead(path));

        // Read header
        string magic = reader.ReadString();
        if (magic != "NNWGT")
            throw new FormatException("Invalid weight file");

        int version = reader.ReadInt32();
        int layerCount = reader.ReadInt32();

        if (layerCount != model.Layers.Count)
            throw new FormatException("Layer count mismatch");

        for (int i = 0; i < layerCount; i++)
        {
            var layer = model.Layers[i];
            string layerName = reader.ReadString();

            int rows = reader.ReadInt32();
            int cols = reader.ReadInt32();

            if (rows > 0 && cols > 0)
            {
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        layer.Weights[r, c] = reader.ReadSingle();
            }

            int biasLen = reader.ReadInt32();
            if (biasLen > 0)
            {
                for (int b = 0; b < biasLen; b++)
                    layer.Biases[b] = reader.ReadSingle();
            }
        }
    }
}
```

## JSON Config Export

```csharp
public class ConfigExporter
{
    public static string ExportConfig(Sequential model)
    {
        var config = new ModelConfig
        {
            Name = "Sequential",
            Layers = model.Layers.Select(l => new LayerConfig
            {
                Type = l.GetType().Name,
                Units = l.Units,
                Activation = l.Activation,
                InputShape = l.InputShape,
                OutputShape = l.OutputShape,
                Parameters = GetLayerParameters(l)
            }).ToList()
        };

        return JsonSerializer.Serialize(config, new JsonSerializerOptions
        {
            WriteIndented = true
        });
    }

    public static Sequential ImportConfig(string json)
    {
        var config = JsonSerializer.Deserialize<ModelConfig>(json);
        var model = new Sequential();

        foreach (var layerConfig in config.Layers)
        {
            var layer = CreateLayer(layerConfig);
            model.Add(layer);
        }

        return model;
    }

    private static Layer CreateLayer(LayerConfig config)
    {
        return config.Type switch
        {
            "Dense" => new Dense(config.Units, activation: config.Activation),
            "LSTM" => new LSTM(config.Units),
            "Embedding" => new Embedding(
                config.Parameters["vocabSize"],
                config.Parameters["embeddingDim"]),
            // ... other layers
            _ => throw new NotSupportedException($"Unknown layer type: {config.Type}")
        };
    }
}
```

## Checkpoint System

```csharp
// Already implemented in NeuralNetwork/Training/Checkpoint.cs
// Review for usage patterns

public class Checkpoint
{
    public static void Save(string path, Sequential model, Optimizer optimizer, int epoch, float loss)
    {
        using var writer = new BinaryWriter(File.Create(path));

        // Metadata
        writer.Write(epoch);
        writer.Write(loss);

        // Model weights
        WeightExporter.ExportWeights(model, writer);

        // Optimizer state
        optimizer.SaveState(writer);
    }

    public static (int epoch, float loss) Load(string path, Sequential model, Optimizer optimizer)
    {
        using var reader = new BinaryReader(File.OpenRead(path));

        int epoch = reader.ReadInt32();
        float loss = reader.ReadSingle();

        WeightExporter.ImportWeights(model, reader);
        optimizer.LoadState(reader);

        return (epoch, loss);
    }
}
```

## Usage Examples

```csharp
// Save trained model (native format)
model.Save("my_model.nn");

// Export weights only
WeightExporter.ExportWeights(model, "weights.bin");

// Export config
string config = ConfigExporter.ExportConfig(model);
File.WriteAllText("model_config.json", config);

// Export to ONNX
var exporter = new OnnxExporter(model);
exporter.Export("model.onnx", new int[] { -1, 128 }); // dynamic batch

// Save checkpoint during training
Checkpoint.Save($"checkpoint_epoch_{epoch}.ckpt", model, optimizer, epoch, loss);

// Resume training from checkpoint
var (resumeEpoch, lastLoss) = Checkpoint.Load("checkpoint.ckpt", model, optimizer);
```

## Related Agents

- Use `/test-model` to verify exported model produces correct outputs
- Use `/architecture` to understand model structure for export
- Use `/benchmark` to compare native vs exported inference speed

## Quality Checklist

- [ ] Weights exported correctly (verified with re-import)
- [ ] Activations included in export
- [ ] Batch dimension handled (dynamic vs fixed)
- [ ] ONNX validated with checker tool
- [ ] Inference outputs match original model
- [ ] File size reasonable
- [ ] Version info included for compatibility
