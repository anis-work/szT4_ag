"""Creates sample PDF resumes and a job requirements file for testing."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

os.makedirs("resumes", exist_ok=True)


def make_pdf(filepath, lines):
    c = canvas.Canvas(filepath, pagesize=A4)
    _, height = A4
    y = height - 50
    for line in lines:
        c.setFont("Helvetica-Bold" if not line.startswith(" ") and line and not line.startswith("-") else "Helvetica", 11)
        c.drawString(50, y, line)
        y -= 16
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()


resumes = {
    "alice_johnson.pdf": [
        "Alice Johnson - Senior Backend Engineer",
        "",
        "EXPERIENCE:",
        "Senior Python Backend Engineer at TechCorp (2 years)",
        "  - Led FastAPI microservices redesign",
        "  - Optimized PostgreSQL queries by 60%",
        "  - AWS: EC2, RDS, Lambda",
        "  - Mentored 3 junior engineers",
        "",
        "Backend Engineer at StartupXYZ (3 years)",
        "  - REST APIs with Django and FastAPI",
        "  - PostgreSQL database design",
        "  - Docker and Kubernetes deployments",
        "",
        "SKILLS: Python, FastAPI, Django, PostgreSQL, AWS, Docker, Kubernetes",
        "EDUCATION: B.S. Computer Science",
    ],
    "david_brown.pdf": [
        "David Brown - Principal Software Engineer",
        "",
        "EXPERIENCE:",
        "Principal Engineer at EnterpriseCorps (5 years)",
        "  - Architect for 50M+ user distributed systems",
        "  - Python and C++ expert",
        "  - Led teams of 8+ engineers",
        "  - Multi-region AWS and GCP deployments",
        "",
        "Senior Software Engineer at MegaTech (4 years)",
        "  - High-throughput system design",
        "  - Database optimization expert",
        "",
        "SKILLS: Python, C++, System Architecture, AWS, GCP, PostgreSQL, Leadership",
        "EDUCATION: M.S. Computer Science",
    ],
    "bob_smith.pdf": [
        "Bob Smith - Mid-Level Backend Developer",
        "",
        "EXPERIENCE:",
        "Backend Developer at WebServices Inc. (2 years)",
        "  - Flask and FastAPI REST APIs",
        "  - Basic PostgreSQL and MySQL",
        "  - Some AWS EC2 experience",
        "",
        "Junior Developer at LocalSoftware (1.5 years)",
        "  - Maintained legacy Python applications",
        "  - Basic database work",
        "",
        "SKILLS: Python, Flask, FastAPI, PostgreSQL, AWS basics, Git",
        "EDUCATION: B.S. Software Engineering",
    ],
}

for filename, lines in resumes.items():
    path = os.path.join("resumes", filename)
    make_pdf(path, lines)
    print(f"Created {path}")

jd_text = """POSITION: Senior Python Backend Engineer

REQUIREMENTS:
- 4+ years of professional Python backend development
- Strong expertise with FastAPI or similar Python web frameworks
- Solid experience with PostgreSQL database design and optimization
- Cloud platform experience (AWS preferred)
- System design and architectural thinking
- Docker and containerization experience
- REST API design principles
- Microservices architecture experience
- Ability to mentor junior developers

PREFERRED:
- Kubernetes experience
- CI/CD pipeline implementation
- Distributed systems experience
"""

with open("resumes/job_requirements.txt", "w") as f:
    f.write(jd_text)
print("Created resumes/job_requirements.txt")
