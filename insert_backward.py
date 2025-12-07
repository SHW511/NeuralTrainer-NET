#!/usr/bin/env python3
"""
Script to insert backward pass methods into TTSModelCuda.cs
"""
import re

# Read the backward methods
with open('backward_methods.txt', 'r') as f:
    backward_code = f.read()
    # Remove the header comments
    backward_code = '\n'.join([line for line in backward_code.split('\n') if not line.strip().startswith('//')])
    backward_code = backward_code.strip() + '\n'

# Read the current TTSModelCuda.cs file
cuda_file = r'NeuralNetwork\Models\TTS\TTSModelCuda.cs'
with open(cuda_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if Backward method already exists
if 'public void Backward(' in content:
    print("Backward() method already exists in TTSModelCuda.cs!")
    print("Skipping insertion.")
else:
    # Find the line after "return (melRefined, stopTokenList.ToArray(), attentionOut);"
    # and before the next method
    pattern = r'(return \(melRefined, stopTokenList\.ToArray\(\), attentionOut\);\s*}\s*)\n(\s*///)'

    replacement = r'\1\n' + backward_code + r'\n\2'

    new_content = re.sub(pattern, replacement, content, count=1)

    if new_content == content:
        print("ERROR: Could not find insertion point!")
        print("Looking for pattern after 'return (melRefined, stopTokenList.ToArray(), attentionOut);'")
    else:
        # Write the modified file
        with open(cuda_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Successfully inserted backward pass methods into {cuda_file}")
        print("Added methods: Backward(), InitializeGradients(), ZeroGradients(),")
        print("               BackwardPostnet(), BackwardDecoder(), ApplyGradients(), ApplyAdamUpdate()")

print("\nDone!")
