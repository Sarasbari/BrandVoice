import os
import re
import json
import random
import argparse
from collections import Counter

# Plausible templates and lists for instruction/input generation
INSTRUCTION_TEMPLATES = {
    "how-to": [
        "Write a how-to guide explaining how to {topic}.",
        "Create a helpful tutorial showing how to {topic}.",
        "Can you write a guide on {topic} for our users?",
        "Draft a step-by-step how-to post on {topic}.",
        "Write an educational post explaining how to achieve {topic}."
    ],
    "announcement": [
        "Write an announcement blog post for {topic}.",
        "Draft a launch post announcing {topic}.",
        "Announce the release of {topic} in an exciting tone.",
        "Can you write a post announcing our new {topic}?",
        "Write a product announcement about the launch of {topic}."
    ],
    "culture": [
        "Write a culture post about {topic}.",
        "Draft a blog post reflecting on {topic}.",
        "Write a post sharing the story behind {topic}.",
        "Can you write a culture blog post discussing {topic}?",
        "Write a piece detailing our team's experience with {topic}."
    ],
    "product": [
        "Write a product blog post about {topic}.",
        "Draft a post outlining the features of {topic}.",
        "Can you write a product post describing {topic}?",
        "Write an overview of {topic} for the blog.",
        "Write a blog post about {topic}."
    ]
}

AUDIENCES = [
    "Notion users",
    "knowledge workers",
    "collaborators",
    "startup founders",
    "engineering teams",
    "product managers",
    "remote workers",
    "business users",
    "designers",
    "creators",
    "managers"
]

TONES = [
    "excited but grounded",
    "thoughtful, not salesy",
    "helpful",
    "clear and direct",
    "professional yet friendly",
    "introspective",
    "excited",
    "informative"
]

def clean_topic_from_title(title):
    # Strip common suffixes and trim whitespace
    topic = title.strip()
    # If the title starts with "How " or "Why ", lowercase the first word for natural flow in template
    if topic.lower().startswith("how ") or topic.lower().startswith("why "):
        words = topic.split(" ")
        words[0] = words[0].lower()
        topic = " ".join(words)
    return topic

def infer_content_type(title):
    title_lower = title.lower()
    if any(w in title_lower for w in ["how to", "guide", "tutorial", "learn", "manage", "template", "create", "build", "use", "setup", "roadmap", "workflow", "steps", "skills", "practices"]):
        return "how-to"
    elif any(w in title_lower for w in ["announcing", "introducing", "new", "launch", "release", "presents", "bring", "expands", "welcome", "introducing"]):
        return "announcement"
    elif any(w in title_lower for w in ["culture", "team", "mission", "story", "history", "meet", "people", "life", "workplace", "intern", "retrospective", "founders", "work styles"]):
        return "culture"
    else:
        return "product"

def split_into_sentences(text):
    # Simple regex sentence splitter matching common sentence terminations
    # Looks for ., ! or ? followed by whitespace and a capital letter or end of string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings
    return [s for s in sentences if s.strip()]

def parse_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    lines = content.splitlines()
    title = ""
    published_date = ""
    url = ""
    body_lines = []
    
    in_body = False
    for line in lines:
        if not in_body:
            if line.startswith("Title: "):
                title = line[len("Title: "):].strip()
            elif line.startswith("Published Date: "):
                published_date = line[len("Published Date: "):].strip()
            elif line.startswith("URL: "):
                url = line[len("URL: "):].strip()
            elif line.strip() == "":
                # First empty line after URL marks the body start
                if url:
                    in_body = True
        else:
            body_lines.append(line)
            
    body_text = "\n".join(body_lines).strip()
    # Split paragraphs by double newlines or multiple newlines
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', body_text) if p.strip()]
    
    return {
        "title": title,
        "published_date": published_date,
        "url": url,
        "paragraphs": paragraphs
    }

def main():
    parser = argparse.ArgumentParser(description="Convert scraped Notion blog posts into instruction-completion JSONL pairs.")
    parser.add_argument("--input-dir", type=str, default=os.path.join("data", "raw", "blog"), help="Directory containing scraped article .txt files")
    parser.add_argument("--output-file", type=str, default=os.path.join("data", "dataset.jsonl"), help="Output path for JSONL dataset")
    parser.add_argument("--validate", action="store_true", help="Perform checks on key presence and output length bounds")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        return
        
    # Get all .txt files
    txt_files = [f for f in os.listdir(args.input_dir) if f.endswith(".txt")]
    print(f"Found {len(txt_files)} scraped article files to process.")
    
    valid_count = 0
    invalid_count = 0
    skipped_count = 0
    
    dataset_records = []
    content_type_counter = Counter()
    
    for filename in txt_files:
        file_path = os.path.join(args.input_dir, filename)
        try:
            parsed = parse_txt_file(file_path)
            paragraphs = parsed["paragraphs"]
            title = parsed["title"]
            
            if not title or not paragraphs:
                skipped_count += 1
                continue
                
            content_type = infer_content_type(title)
            topic = clean_topic_from_title(title)
            
            article_pairs = []
            
            # 1. Opening paragraph pair (blog intro)
            intro_candidate = paragraphs[0]
            # Check length bounds (60-500 chars) for intro
            if 60 <= len(intro_candidate) <= 500 and "http" not in intro_candidate:
                # Valid intro
                audience = random.choice(AUDIENCES)
                tone = random.choice(TONES)
                # Formulate instruction
                templates = INSTRUCTION_TEMPLATES[content_type]
                instruction = random.choice(templates).format(topic=topic)
                
                pair = {
                    "instruction": instruction,
                    "input": f"Topic: {topic}. Audience: {audience}. Tone: {tone}",
                    "output": intro_candidate
                }
                article_pairs.append(pair)
            else:
                skipped_count += 1  # Count skipped intro
                
            # 2. Body paragraph pairs
            # We want to select additional paragraphs to reach a target of 3 to 5 pairs in total
            # So we look for up to 4 more qualifying paragraphs
            for para in paragraphs[1:]:
                # If we already have 5 pairs, we can stop
                if len(article_pairs) >= 5:
                    break
                    
                # Skip rule: list markers
                if para.startswith(("-", "*", "•")) or re.match(r'^\d+\.\s', para):
                    skipped_count += 1
                    continue
                    
                # Skip rule: contains URLs
                if "http://" in para or "https://" in para or "www." in para:
                    skipped_count += 1
                    continue
                    
                # Skip rule: ends with no punctuation (likely a subheader or header)
                if not para[-1] in [".", "!", "?", '"', "'", "”", "’"]:
                    skipped_count += 1
                    continue
                    
                # Skip rule: wrong length
                if len(para) < 60 or len(para) > 500:
                    skipped_count += 1
                    continue
                    
                # Skip rule: must be 2-4 sentences
                sentences = split_into_sentences(para)
                if not (2 <= len(sentences) <= 4):
                    skipped_count += 1
                    continue
                    
                # If it passes all criteria, formulate pair!
                audience = random.choice(AUDIENCES)
                tone = random.choice(TONES)
                templates = INSTRUCTION_TEMPLATES[content_type]
                instruction = random.choice(templates).format(topic=topic)
                
                pair = {
                    "instruction": instruction,
                    "input": f"Topic: {topic}. Audience: {audience}. Tone: {tone}",
                    "output": para
                }
                article_pairs.append(pair)
                
            # If the article yields fewer than 3 pairs, the PRD specifies we extract 3-5 pairs.
            # But what if there are simply not enough qualifying paragraphs in the text?
            # We should save what we found, but if we need to hit 3, we can relax the sentence constraint
            # slightly or just write the 1 or 2 we found. Let's write whatever we found.
            for pair in article_pairs:
                # Validation checks on the fly (or if we save it)
                # Checking keys exist
                if not all(k in pair for k in ["instruction", "input", "output"]):
                    invalid_count += 1
                # Checking output length bounds
                elif not (60 <= len(pair["output"]) <= 500):
                    invalid_count += 1
                else:
                    valid_count += 1
                    dataset_records.append(pair)
                    content_type_counter[content_type] += 1
                    
        except Exception as e:
            # If parsing of file fails, log it or count it
            invalid_count += 1
            
    # Save output to dataset.jsonl
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for record in dataset_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
    print("\n" + "="*40)
    print("JSONL Build Completed.")
    print(f"Saved dataset to: {args.output_file}")
    
    # Print content-type breakdown
    print("\nContent-Type Breakdown:")
    for ct, count in content_type_counter.items():
        print(f"  - {ct}: {count}")
    print(f"  - Total Samples: {len(dataset_records)}")
    
    # If validate flag is passed or by default, print validation summary in the exact format
    if args.validate:
        print("\nValidation Summary:")
        # The prompt requires: Prints: 'Valid: 214 | Invalid: 3 | Skipped: 8' (with single quotes or just matching text)
        print(f"Valid: {valid_count} | Invalid: {invalid_count} | Skipped: {skipped_count}")
    print("="*40)

if __name__ == "__main__":
    main()
