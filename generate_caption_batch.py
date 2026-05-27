from generate_caption import batch_generate

#batch_generate("tasks_final.json", "tasks_with_caption.json")          # 전체
batch_generate("final task_eng.json", "tasks_with_caption.json", limit=5) # 5개만 테스트

