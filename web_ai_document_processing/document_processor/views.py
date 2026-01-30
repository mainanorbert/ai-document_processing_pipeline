from django.shortcuts import render

# Create your views here.
import os
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage

from document_processor.services.llm import run_llm_document_extraction

from django.views.decorators.csrf import csrf_exempt

@csrf_exempt
@require_POST
def process_document(request):
    """
    Accept an uploaded image and return extracted data.
    """
    uploaded_file = request.FILES.get("document")

    if not uploaded_file:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    file_path = default_storage.save(
        f"uploads/{uploaded_file.name}", uploaded_file
    )
    print(f"File saved to: {file_path}")

    absolute_path = os.path.join(default_storage.location, file_path)

    result = run_llm_document_extraction(image_path=absolute_path)

    return JsonResponse(result, status=200)

def index(request):
    """
    Render the main upload interface.
    """
    return render(request, 'document_processor/index.html')