# Quick Reference Guide

## File Organization

### Core Application
```
app.py (90 lines)
├── Imports utility modules
├── Configures Streamlit page
├── Renders UI components
├── Handles user interactions
└── Orchestrates workflow
```

### Utils Package
```
utils/
├── ui_components.py (300+ lines)
│   ├── CSS styles
│   ├── Header rendering
│   ├── Form components
│   ├── Table display
│   ├── Filters & sorting
│   └── Export functionality
│
├── pipeline.py (180+ lines)
│   ├── Kernel initialization
│   ├── File upload handling
│   ├── CV text extraction
│   ├── Embedding generation
│   ├── Vector search
│   ├── AI ranking
│   └── Retry logic
│
└── results_handler.py (30 lines)
    └── Results display orchestration
```

## Common Tasks

### Adding a New UI Component
**File**: `utils/ui_components.py`
```python
def render_my_component():
    """Render my new component."""
    st.markdown('<div class="my-class">', unsafe_allow_html=True)
    # UI code here
    st.markdown('</div>', unsafe_allow_html=True)
```

### Modifying the AI Pipeline
**File**: `utils/pipeline.py`
```python
async def run_pipeline(kernel, cvs, jd, status_placeholder):
    #pipeline modifications here
    pass
```

### Changing Styles
**File**: `utils/ui_components.py`
```python
def apply_custom_styles():
    st.markdown("""
    <style>
        /* Add your CSS here */
    </style>
    """, unsafe_allow_html=True)
```

### Adding a New Filter
**File**: `utils/ui_components.py`
```python
def render_filters():
    # Add new filter controls
    new_filter = st.selectbox("New Filter", options)
    return min_score, sort_by, new_filter
```

## Import Guide

### In app.py
```python
from utils.ui_components import (
    apply_custom_styles,
    render_header,
    render_file_upload_section,
    # ... other UI functions
)
from utils.pipeline import (
    get_kernel,
    save_uploads,
    build_cvs,
    run_pipeline
)
from utils.results_handler import show_results
```

### In other modules
```python
# UI components can import from each other
from utils.ui_components import render_stats_bar

# Pipeline can use models
from models import CV, JobDescription, RankedResult
```

## Testing Strategy

### Unit Tests
```python
# Test UI components (mock Streamlit)
def test_render_stats_bar():
    results = [mock_result1, mock_result2]
    # Test rendering logic

# Test pipeline functions
def test_build_cvs():
    tmp_dir = "/tmp/test"
    filenames = ["resume1.pdf"]
    cvs, skipped = build_cvs(tmp_dir, filenames)
    assert len(cvs) > 0
```

### Integration Tests
```python
# Test full pipeline
async def test_full_pipeline():
    kernel = get_kernel()
    cvs = [mock_cv1, mock_cv2]
    jd = JobDescription(role="Engineer", requirements="...")
    results = await run_pipeline(kernel, cvs, jd, mock_placeholder)
    assert len(results) == 2
```

## Debugging Tips

### UI Issues
1. Check `utils/ui_components.py`
2. Verify CSS classes in `apply_custom_styles()`
3. Check Streamlit component parameters

### Pipeline Issues
1. Check `utils/pipeline.py`
2. Add logging statements
3. Verify API keys in `.env`
4. Check retry logic in `_invoke_with_retry()`

### Results Display Issues
1. Check `utils/results_handler.py`
2. Verify filter logic in `apply_filters_and_sort()`
3. Check dataframe column configuration

## Performance Optimization

### Caching
```python
# Already implemented in pipeline.py
@st.cache_resource
def get_kernel() -> Kernel:
    # Kernel is cached across reruns
    pass
```

### Async Operations
```python
# Pipeline uses async for better performance
async def run_pipeline(...):
    # Async operations here
    pass
```

## Code Style

### Naming Conventions
- **Functions**: `snake_case` (e.g., `render_header()`)
- **Classes**: `PascalCase` (e.g., `RankedResult`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `RANKING_PROMPT`)
- **Private functions**: `_leading_underscore` (e.g., `_invoke_with_retry()`)

### Documentation
- All functions have docstrings
- Complex logic has inline comments
- Module-level docstrings explain purpose

### Type Hints
```python
def render_filters() -> tuple[int, str]:
    """Render filter controls."""
    pass

def show_results(results: list, role: str, cvs: list) -> None:
    """Display results."""
    pass
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd ag_sz__t4

# Run the app
streamlit run app.py
```

### Module Not Found
```python
# Check if __init__.py exists in utils/
# Verify Python path includes project root
```

### Streamlit Errors
```python
# Clear cache
streamlit cache clear

# Restart app
# Press 'R' in browser or Ctrl+C and restart
```

## Quick Commands

```bash
# Run the application
streamlit run app.py

# Run with specific port
streamlit run app.py --server.port 8502

# Clear cache and run
streamlit cache clear && streamlit run app.py

# Check Python imports
python -c "from utils import ui_components; print('OK')"
```

## Summary

**Remember**:
- 📁 `app.py` = Orchestration
- 🎨 `ui_components.py` = Visual layer
- ⚙️ `pipeline.py` = Business logic
- 📊 `results_handler.py` = Results display

**Golden Rule**: Keep concerns separated!
