#!/usr/bin/env python3
"""
Convert the 1k_training.jsonl from messages format to Mixtral instruction format
"""

import json
import os

def convert_messages_to_mixtral_format(input_file, output_file):
    """Convert messages format to Mixtral [INST] format"""
    
    converted_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if line.strip():
                data = json.loads(line)
                messages = data['messages']
                
                # Extract user and assistant messages
                user_message = messages[0]['content']
                assistant_message = messages[1]['content']
                
                # Clean up Unicode escape sequences
                user_message = user_message.replace('\u2019', "'").replace('\u2013', '-').replace('\u2014', '--')
                assistant_message = assistant_message.replace('\u2019', "'").replace('\u2013', '-').replace('\u2014', '--')
                
                # Convert to Mixtral instruction format
                mixtral_text = f"[INST] {user_message} [/INST] {assistant_message}"
                
                # Create new format with 'text' field
                new_data = {"text": mixtral_text}
                
                # Write to output file (ensure_ascii=False to preserve Unicode properly)
                outfile.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                converted_count += 1
    
    print(f"✅ Converted {converted_count} examples to Mixtral format")
    print(f"📁 Output saved to: {output_file}")

def main():
    input_file = "1k_training.jsonl"
    output_file = "1k_training_mixtral.jsonl"
    
    if not os.path.exists(input_file):
        print(f"❌ Input file {input_file} not found!")
        return
    
    print(f"🔄 Converting {input_file} to Mixtral format...")
    convert_messages_to_mixtral_format(input_file, output_file)
    
    # Show sample of converted data
    print("\n📋 Sample converted examples:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 2:  # Show first 2 examples
                data = json.loads(line)
                print(f"\nExample {i+1}:")
                print(f"Text: {data['text'][:100]}...")
            else:
                break

if __name__ == "__main__":
    main()
