import csv
import json

def process_qa_csv(input_file, questions_output, metadata_output):
    questions = []
    qa_metadata = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            question = row['Question']
            answer = row['Answer']
            questions.append(question)
            qa_metadata[str(idx)] = {
                'question': question,
                'answer': answer,
                'source_doc_ids': []
            }
    
    with open(questions_output, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(f'{q}\n')
    
    with open(metadata_output, 'w', encoding='utf-8') as f:
        json.dump(qa_metadata, f, indent=2)
    
    print(f'Processed {len(questions)} questions from {input_file}')
    print(f'Questions saved to {questions_output}')
    print(f'Metadata saved to {metadata_output}')

if __name__ == '__main__':
    process_qa_csv('official_order_with_answers.csv', 'official_test.txt', 'qa_metadata.json')