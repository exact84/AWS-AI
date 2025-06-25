const form = document.getElementById('qa-form');
const questionInput = document.getElementById('question');
const answerDiv = document.getElementById('answer');
const sourcesDiv = document.getElementById('sources');
const loadingDiv = document.getElementById('loading');
const errorDiv = document.getElementById('error');

form.onsubmit = async (e) => {
  e.preventDefault();
  answerDiv.style.display = 'none';
  sourcesDiv.style.display = 'none';
  errorDiv.style.display = 'none';
  loadingDiv.style.display = '';
  answerDiv.textContent = '';
  sourcesDiv.textContent = '';
  errorDiv.textContent = '';
  try {
    const question = questionInput.value.trim();
    const res = await fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    loadingDiv.style.display = 'none';
    if (!res.ok) {
      errorDiv.textContent = data.error || 'Error occurred';
      errorDiv.style.display = '';
      return;
    }
    answerDiv.textContent = data.answer;
    answerDiv.style.display = '';
    if (data.sources && data.sources.length) {
      sourcesDiv.innerHTML =
        '<div class="source-title">Sources:</div>' +
        data.sources
          .map(
            (s, i) =>
              `<div class="source-block"><b>[${i + 1}]</b> (Object ID: ${s.objectId}, Chunk: ${s.chunkIndex})<br>${s.text.replace(/\n/g, '<br>')}</div>`
          )
          .join('');
      sourcesDiv.style.display = '';
    }
  } catch (err) {
    loadingDiv.style.display = 'none';
    errorDiv.textContent = 'Network or server error.';
    errorDiv.style.display = '';
  }
};
