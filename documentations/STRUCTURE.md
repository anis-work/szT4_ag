# Project Structure Documentation

## Overview
The CV Ranking Agent codebase has been refactored to separate UI components from business logic, making the code more maintainable, testable, and organized.

## New Directory Structure

```
ag_sz__t4/
├── app.py                          # Main Streamlit application (simplified)
├── config.py                       # Configuration loading
├── models.py                       # Pydantic data models
├── embedder.py                     # Text embedding functions
├── vector_store.py                 # Vector storage and similarity search
├── pdf_loader.py                   # PDF/DOCX text extraction
├── validator.py                    # Result validation
├── plugins/
│   ├── cv_ingestion_plugin.py      # CV ingestion plugin
│   ├── cv_retrieval_plugin.py      # CV retrieval plugin
│   └── cv_ranking_plugin.py        # CV ranking plugin
└── utils/                          # NEW: Utility modules
    ├── __init__.py                 # Package initialization
    ├── ui_components.py            # UI rendering functions
    ├── pipeline.py                 # Business logic and pipeline
    └── results_handler.py          # Results display logic
```

## Module Breakdown

### 1. `app.py` (Main Application)
**Purpose**: Entry point for the Streamlit application
**Responsibilities**:
- Page configuration
- Session state initialization
- Orchestrating UI components
- Handling user interactions
- Minimal business logic

**Key Functions**:
- Imports and uses utility modules
- Coordinates between UI and business logic
- Manages application flow

---

### 2. `utils/ui_components.py` (UI Layer)
**Purpose**: All UI rendering and styling functions
**Responsibilities**:
- Custom CSS styles
- Component rendering (header, forms, tables, etc.)
- User input handling
- Visual presentation

**Key Functions**:
- `apply_custom_styles()` - Apply CSS to the app
- `render_header()` - Display main header
- `render_file_upload_section()` - File upload UI
- `render_job_details_section()` - Job input form
- `render_action_button()` - Analyze button with status
- `render_stats_bar()` - Statistics display
- `render_filters()` - Filter controls
- `render_results_table()` - Results table display
- `render_export_section()` - CSV download section
- `render_history_section()` - Analysis history UI
- `apply_filters_and_sort()` - Filter and sort logic

---

### 3. `utils/pipeline.py` (Business Logic Layer)
**Purpose**: Core business logic and AI pipeline
**Responsibilities**:
- Kernel initialization
- File processing
- CV extraction and parsing
- AI ranking pipeline execution
- Error handling and retries

**Key Functions**:
- `get_kernel()` - Initialize Semantic Kernel with Gemini
- `save_uploads()` - Save uploaded files securely
- `build_cvs()` - Extract text and create CV objects
- `_invoke_with_retry()` - Retry logic for API calls
- `run_pipeline()` - Complete ranking pipeline

**Constants**:
- `RANKING_PROMPT` - AI prompt template for ranking

---

### 4. `utils/results_handler.py` (Results Layer)
**Purpose**: Handle results display and export
**Responsibilities**:
- Coordinate results display
- Apply filters and sorting
- Manage export functionality

**Key Functions**:
- `show_results()` - Main results display orchestrator

---

## Benefits of This Structure

### 1. **Separation of Concerns**
- UI code is isolated from business logic
- Each module has a single, clear responsibility
- Easier to understand and maintain

### 2. **Reusability**
- UI components can be reused across different views
- Business logic can be tested independently
- Pipeline functions can be used in CLI or other interfaces

### 3. **Testability**
- UI components can be tested separately
- Business logic can be unit tested without Streamlit
- Mock data can be easily injected

### 4. **Maintainability**
- Changes to UI don't affect business logic
- Changes to AI pipeline don't affect UI
- Easier to locate and fix bugs

### 5. **Scalability**
- Easy to add new UI components
- Simple to extend business logic
- Can add new features without touching existing code

---

## How It Works

### Application Flow

```
User Input (app.py)
    ↓
UI Components (ui_components.py)
    ↓
Business Logic (pipeline.py)
    ↓
Results Handler (results_handler.py)
    ↓
UI Display (ui_components.py)
```

### Example: Running Analysis

1. **User clicks "Analyze Candidates"** (`app.py`)
2. **Files are saved** (`pipeline.save_uploads()`)
3. **CVs are built** (`pipeline.build_cvs()`)
4. **Pipeline runs** (`pipeline.run_pipeline()`)
   - Embeddings generated
   - Vector search performed
   - AI ranking executed
5. **Results displayed** (`results_handler.show_results()`)
   - Stats bar rendered (`ui_components.render_stats_bar()`)
   - Filters applied (`ui_components.render_filters()`)
   - Table shown (`ui_components.render_results_table()`)
   - Export enabled (`ui_components.render_export_section()`)

---

## Migration Notes

### What Changed
- **app.py**: Reduced from ~470 lines to ~90 lines
- **New files**: 3 utility modules created
- **Functionality**: 100% preserved, zero breaking changes

### What Stayed the Same
- All features work exactly as before
- No changes to models, config, or plugins
- Same user experience
- Same API integrations

---

## Future Enhancements

With this structure, you can easily:

1. **Add new UI themes** - Modify only `ui_components.py`
2. **Switch AI providers** - Modify only `pipeline.py`
3. **Add CLI interface** - Reuse `pipeline.py` functions
4. **Add unit tests** - Test each module independently
5. **Add new features** - Create new utility modules
6. **Optimize performance** - Profile and optimize specific modules

---

## Best Practices

### When to modify each file:

- **app.py**: Application flow, session state, page config
- **ui_components.py**: Visual changes, new UI elements, styling
- **pipeline.py**: AI logic, file processing, API calls
- **results_handler.py**: Results display logic, filtering, sorting

### Guidelines:

1. Keep `app.py` minimal - it should orchestrate, not implement
2. UI components should not contain business logic
3. Pipeline functions should not render UI
4. Each function should do one thing well
5. Use type hints for better code clarity

---

## Summary

The refactored structure provides:
- ✅ Clean separation of UI and logic
- ✅ Better code organization
- ✅ Easier maintenance and testing
- ✅ Same functionality as before
- ✅ Foundation for future enhancements
