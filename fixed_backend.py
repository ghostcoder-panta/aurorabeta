"""Backend para o Aurora: Cria um servidor local em
localhost:5000; apresenta a página index.html e procura os
dados na base clinical_trials.csv.
TO DO LIST: filtro para países não funciona.

Autor: Marcos Pantarotto (+ AI)
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)


# Configurable CSV file path - checks multiple common locations
def find_csv_file():
    """Find the CSV file in common locations"""
    possible_paths = [
        'clinical_trials.csv',  # Same directory as app.py
        './data/clinical_trials.csv',  # data subdirectory
        './clinical_trials.csv',  # explicit current directory
        os.path.join(os.path.dirname(__file__), 'clinical_trials.csv'),  # same dir as script
        os.path.join(os.path.dirname(__file__), 'data', 'clinical_trials.csv'),  # data subdir relative to script
        # Add more common paths here instead of hardcoded user path
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found CSV file at: {path}")
            return path

    # Check current working directory and subdirectories
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == 'clinical_trials.csv':
                found_path = os.path.join(root, file)
                print(f"Found CSV file at: {found_path}")
                return found_path

    print("CSV file not found in any expected location")
    print("Please ensure 'clinical_trials.csv' is in the same directory as app.py or in a 'data' subdirectory")
    return None


CSV_FILE_PATH = find_csv_file()

# Cache for the dataframe with timestamp
cache = {
    'df': None,
    'last_loaded': None,
    'file_mtime': None
}


def load_csv():
    """Load CSV file with caching to improve performance"""
    required_columns = [
        'nct_id', 'official_title', 'acronym', 'phase',
        'countries', 'locations', 'condition', 'num_arms',
        'arms_description', 'inclusion_criteria', 'exclusion_criteria'
    ]

    if not CSV_FILE_PATH:
        print("No CSV file path available")
        return pd.DataFrame(columns=required_columns)

    try:
        file_mtime = os.path.getmtime(CSV_FILE_PATH)

        if cache['df'] is None or cache['file_mtime'] != file_mtime:
            print(f"Loading CSV from: {CSV_FILE_PATH}")
            df = pd.read_csv(CSV_FILE_PATH, encoding='utf-8')

            # Add missing columns with empty strings
            for col in required_columns:
                if col not in df.columns:
                    df[col] = ''

            # Fill NaN values
            df = df.fillna('')

            cache['df'] = df
            cache['last_loaded'] = datetime.now()
            cache['file_mtime'] = file_mtime

            print(f"CSV loaded successfully: {len(df)} studies found")

        return cache['df']

    except FileNotFoundError:
        print(f"CSV not found at {CSV_FILE_PATH}")
        return pd.DataFrame(columns=required_columns)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame(columns=required_columns)


def categorize_phases(phase_str):
    """Categorize phase into multiple standard groups if applicable"""
    if pd.isna(phase_str) or phase_str == '':
        return ['Outras']

    phase_lower = str(phase_str).lower()
    phases = []

    # Check for each phase independently
    if 'phase1' in phase_lower or 'phase 1' in phase_lower or 'early phase1' in phase_lower:
        phases.append('Fase 1')
    if 'phase2' in phase_lower or 'phase 2' in phase_lower:
        phases.append('Fase 2')
    if 'phase3' in phase_lower or 'phase 3' in phase_lower:
        phases.append('Fase 3')
    if 'phase4' in phase_lower or 'phase 4' in phase_lower:
        phases.append('Fase 4')

    # If no standard phases found, categorize as "Outras"
    if not phases:
        if 'na' in phase_lower or phase_lower == 'na':
            phases = ['Outras']
        else:
            phases = ['Outras']

    return phases

def categorize_tumor(condition_str):
    """Categorize condition into tumor types"""
    if pd.isna(condition_str) or condition_str == '':
        return 'Other'

    condition_lower = str(condition_str).lower()

    tumor_categories = {
        'Lung Cancer': ['lung', 'pulmon', 'nsclc', 'sclc', 'pulmonary'],
        'Breast Cancer': ['breast', 'mama', 'mammary'],
        'GI Cancer': ['gastro', 'colon', 'rectal', 'colorectal', 'pancrea',
                      'stomach', 'esophag', 'hepato', 'liver', 'intestin'],
        'Prostate Cancer': ['prostate', 'prÃƒÂ³stata', 'prostatic'],
        'Bladder Cancer': ['bladder', 'bexiga', 'urothel'],
        'Kidney Cancer': ['kidney', 'renal', 'rim', 'nephr'],
        'Blood Cancer': ['leukemia', 'lymphoma', 'myeloma', 'blood', 'hematologic'],
        'Skin Cancer': ['melanoma', 'skin', 'dermat'],
        'Brain Cancer': ['brain', 'glioma', 'neurolog', 'cerebr']
    }

    for category, keywords in tumor_categories.items():
        if any(keyword in condition_lower for keyword in keywords):
            return category

    return 'Other'

def extract_countries(countries_str, locations_str):
    """Extract unique countries from countries and locations fields"""
    countries = set()

    # From countries field
    if countries_str and str(countries_str) != 'nan':
        for country in str(countries_str).split(','):
            country = country.strip()
            if country:
                countries.add(country)

    # From locations field (backup)
    if locations_str and str(locations_str) != 'nan' and not countries:
        location_str = str(locations_str).lower()
        country_patterns = {
            'Portugal': ['portugal', 'lisbon', 'lisboa', 'porto', 'coimbra'],
            'Brazil': ['brazil', 'brasil', 'sÃƒÂ£o paulo', 'rio de janeiro'],
            'Spain': ['spain', 'espaÃƒÂ±a', 'madrid', 'barcelona'],
            'France': ['france', 'paris', 'lyon'],
            'Germany': ['germany', 'deutschland', 'berlin', 'munich'],
            'Italy': ['italy', 'italia', 'rome', 'milan'],
            'United States': ['united states', 'usa', 'u.s.'],
            'United Kingdom': ['united kingdom', 'uk', 'england', 'london']
        }

        for country, patterns in country_patterns.items():
            if any(pattern in location_str for pattern in patterns):
                countries.add(country)

    return list(countries)

@app.route('/')
def index():
    """Serve the main HTML page"""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_html(filename):
    """Serve HTML files from the root directory"""
    if filename.endswith('.html'):
        try:
            return send_from_directory('.', filename)
        except FileNotFoundError:
            return jsonify({'error': 'Page not found'}), 404
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/api/status')
def status():
    """Check API and data status"""
    df = load_csv()
    return jsonify({
        'status': 'ok',
        'csv_loaded': not df.empty,
        'total_studies': len(df),
        'csv_path': CSV_FILE_PATH,
        'last_loaded': cache['last_loaded'].isoformat() if cache['last_loaded'] else None
    })


@app.route('/api/stats')
def get_stats():
    """Get overall statistics for the sidebar"""
    try:
        df = load_csv()
        if df.empty:
            return jsonify({
                'total_studies': 0,
                'total_countries': 0,
                'phase_counts': {},
                'tumor_counts': {}
            })

        # Calculate phase statistics (count each study in all its phases)
        phase_counts = {'Fase 1': 0, 'Fase 2': 0, 'Fase 3': 0, 'Fase 4': 0, 'Outras': 0}
        for _, row in df.iterrows():
            phases = categorize_phases(row['phase'])
            for phase in phases:
                if phase in phase_counts:
                    phase_counts[phase] += 1

        # Calculate tumor statistics
        df['tumor_category'] = df['condition'].apply(categorize_tumor)
        tumor_counts = df['tumor_category'].value_counts().to_dict()

        # Calculate countries
        all_countries = set()
        for _, row in df.iterrows():
            countries = extract_countries(row['countries'], row['locations'])
            all_countries.update(countries)

        return jsonify({
            'total_studies': len(df),
            'total_countries': len(all_countries),
            'phase_counts': phase_counts,
            'tumor_counts': tumor_counts
        })

    except Exception as e:
        print(f"Error in stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_studies():
    """Search studies based on filters"""
    try:
        df = load_csv()
        if df.empty:
            return jsonify({'studies': [], 'total': 0, 'error': 'No data available'})

        data = request.get_json(silent=True) or {}
        phase_filter = data.get('phase', 'all')
        tumor_filter = data.get('tumor', 'all')
        country_filter = data.get('country', 'all')
        title_filter = (data.get('title', '') or '').lower()

        filtered_df = df.copy()

        # Country filter
        if country_filter and country_filter != 'all':
            mask = filtered_df.apply(
                lambda row: (country_filter in str(row['countries']) or
                             country_filter.lower() in str(row['locations']).lower()),
                axis=1
            )
            filtered_df = filtered_df[mask]

        # Phase filter - now checks if the selected phase is in any of the study's phases
        if phase_filter and phase_filter != 'all':
            print(f"Applying phase filter: {phase_filter}")

            # Apply filter checking if phase_filter is in any of the study's phases
            mask = filtered_df['phase'].apply(
                lambda x: phase_filter in categorize_phases(x)
            )

            before_count = len(filtered_df)
            filtered_df = filtered_df[mask]
            after_count = len(filtered_df)
            print(f"Phase filter: {before_count} -> {after_count} studies")

        # Title filter
        if title_filter:
            mask = filtered_df['official_title'].str.lower().str.contains(title_filter, na=False, regex=False)
            filtered_df = filtered_df[mask]

        # Tumor category filter
        filtered_df['tumor_category'] = filtered_df['condition'].apply(categorize_tumor)
        if tumor_filter and tumor_filter != 'all':
            filtered_df = filtered_df[filtered_df['tumor_category'] == tumor_filter]

        # Build results
        results = []
        for _, row in filtered_df.iterrows():
            # Fix primary_country to always return a string
            primary_country = ""
            if row["countries"] and str(row["countries"]) != 'nan':
                parts = [p.strip() for p in re.split(r'[|,]', str(row["countries"])) if p.strip()]
                primary_country = parts[0] if parts else ""

            # Get all phases for this study
            phases_list = categorize_phases(row["phase"])
            phase_display = " + ".join(phases_list)  # Display as "Fase 1 + Fase 2" for multi-phase studies

            study = {
                "nct_id": str(row["nct_id"]) if row["nct_id"] else "",
                "title": str(row["official_title"]) if row["official_title"] else "",
                "phase": phase_display,  # Now shows all phases
                "phases": phases_list,   # Array of phases for filtering
                "tumor_category": categorize_tumor(row["condition"]),
                "conditions": str(row["condition"]) if row["condition"] else "",
                "countries": str(row["countries"]) if row["countries"] else "",
                "primary_country": primary_country,
                "locations": str(row["locations"]) if row["locations"] else "",
                "arms_description": str(row["arms_description"]) if row["arms_description"] else "",
                "inclusion_criteria": str(row["inclusion_criteria"]) if row["inclusion_criteria"] else "",
                "exclusion_criteria": str(row["exclusion_criteria"]) if row["exclusion_criteria"] else "",
                "status": "Recrutando"  # Add default status or derive from data
            }
            results.append(study)

        print(f"Search completed: {len(results)} studies found")
        return jsonify({"studies": results, "total": len(results)})

    except Exception as e:
        print(f"Error in search: {e}")
        return jsonify({'error': str(e), 'studies': [], 'total': 0}), 500

@app.route('/api/filters', methods=['GET'])
def get_filters():
    """Get available filter options"""
    try:
        df = load_csv()
        if df.empty:
            return jsonify({
                'countries': [],
                'phases': ['Fase 1', 'Fase 2', 'Fase 3', 'Fase 4', 'Outras'],
                'tumors': ['Lung Cancer', 'Breast Cancer', 'GI Cancer', 'Prostate Cancer',
                           'Bladder Cancer', 'Kidney Cancer', 'Blood Cancer', 'Skin Cancer',
                           'Brain Cancer', 'Other']
            })

        # Extract unique countries
        all_countries = set()
        for _, row in df.iterrows():
            countries = extract_countries(row['countries'], row['locations'])
            all_countries.update(countries)

        # Extract tumor categories dynamically
        df['tumor_category'] = df['condition'].apply(categorize_tumor)
        all_tumors = sorted([t for t in df['tumor_category'].unique().tolist() if t])

        # Extract phases dynamically
        df['phase_category'] = df['phase'].apply(categorize_phase)
        all_phases = sorted([p for p in df['phase_category'].unique().tolist() if p])

        return jsonify({
            'countries': sorted([c for c in all_countries if c]),
            'phases': all_phases if all_phases else ['Fase 1', 'Fase 2', 'Fase 3', 'Fase 4', 'Outras'],
            'tumors': all_tumors if all_tumors else ['Other']
        })

    except Exception as e:
        print(f"Error in filters: {e}")
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("=" * 50)
    print("Aurora Trials API Server")
    print("=" * 50)

    # Check if CSV file exists at startup
    if CSV_FILE_PATH:
        df = load_csv()
        print(f"Loaded {len(df)} studies from CSV")
    else:
        print("WARNING: No CSV file found. API will return empty results.")
        print("Please ensure 'clinical_trials.csv' is in the same directory as app.py")

    # Get port from environment (Render sets this automatically)
    port = int(os.environ.get('PORT', 5000))

    print(f"Starting server on port {port}")
    print("API endpoints available:")
    print("  GET  /api/status   - Check server and data status")
    print("  GET  /api/stats    - Get overall statistics")
    print("  GET  /api/filters  - Get available filter options")
    print("  POST /api/search   - Search studies with filters")
    print("=" * 50)

    # Production settings for deployment
    app.run(
        debug=False,  # Disable debug in production
        port=port,  # Use environment port or default 5000
        host='0.0.0.0'  # Bind to all interfaces
    )