import asyncio
from src.retrieval.rag_pipeline import RAGPipeline
from src.generation.response_generator import ResponseGenerator
from src.memory.conversation_memory import ConversationMemory

memory = ConversationMemory()
pipeline = RAGPipeline()
generator = ResponseGenerator()

def simulate_turn(pregunta):
    print(f"Pregunta: {pregunta}")
    history = memory.get_history() if not memory.is_empty() else None
    rag_out = pipeline.run(pregunta, k=10, max_documents=5, history=history)
    res = generator.generate(rag_out, history=history)
    print("Respuesta:\n", res["respuesta"])
    memory.add_turn(pregunta, res["respuesta"])
    print("-" * 50)

simulate_turn("que dice la ley sobre el uso de casco en motocicleta")
simulate_turn("si lo llevo desabrochado?")
