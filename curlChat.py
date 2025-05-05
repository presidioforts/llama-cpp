curl http://localhost:8000/chat/completions \
 -H "Content-Type: application/json" \
 -d '{
       "model": "llama-3-finetuned",
       "messages": [
         {"role":"system","content":"You are a helpful assistant."},
         {"role":"user","content":"Convert <h1>Hello</h1> to Markdown."}
       ],
       "max_tokens": 128
     }'
