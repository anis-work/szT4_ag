"""Sample data module for CV Ranking Agent.

Provides realistic sample CVs and job description for testing.
"""

from typing import List
from models import CV, JobDescription


def get_sample_cvs() -> List[CV]:
    """Get sample candidate CVs.
    
    Returns 5 fictional but realistic software engineering CVs with varying
    seniority levels, tech stacks, and experience.
    
    Returns:
        List of CV objects.
    """
    cvs = [
        CV(
            id="cv_001",
            candidate_name="Alice Johnson",
            raw_text="""
Alice Johnson
Senior Backend Engineer

EXPERIENCE:
- Senior Python Backend Engineer at TechCorp (2 years)
  * Led microservices architecture redesign using FastAPI
  * Optimized PostgreSQL database queries, 60% improvement
  * Deployed infrastructure on AWS (EC2, RDS, Lambda)
  * Mentored 3 junior engineers
  
- Backend Engineer at StartupXYZ (3 years)
  * Built REST APIs using Django and FastAPI
  * Designed and maintained PostgreSQL databases
  * Managed deployment pipelines with Docker and Kubernetes
  
SKILLS: Python, FastAPI, Django, PostgreSQL, AWS, Docker, Kubernetes, System Design
EDUCATION: B.S. Computer Science

Strong match for senior backend role with extensive FastAPI and PostgreSQL experience.
            """
        ),
        CV(
            id="cv_002",
            candidate_name="Bob Smith",
            raw_text="""
Bob Smith
Mid-Level Backend Developer

EXPERIENCE:
- Backend Developer at WebServices Inc. (2 years)
  * Developed REST APIs using Flask and FastAPI
  * Basic PostgreSQL and MySQL database work
  * Some AWS experience with EC2
  * Participated in code reviews
  
- Junior Developer at LocalSoftware (1.5 years)
  * Maintained legacy Python applications
  * Database optimization basics
  
SKILLS: Python, Flask, FastAPI, SQL, PostgreSQL, AWS basics, Git
EDUCATION: B.S. Software Engineering

Good candidate but less senior than required, limited cloud experience.
            """
        ),
        CV(
            id="cv_003",
            candidate_name="Carol Williams",
            raw_text="""
Carol Williams
Full Stack Developer

EXPERIENCE:
- Full Stack Developer at WebStartup (2.5 years)
  * Frontend: React, TypeScript, HTML/CSS
  * Backend: Node.js and Express
  * Database: MongoDB and PostgreSQL basics
  * AWS and Azure cloud exposure
  
- Junior Full Stack Developer at AgencyWeb (1 year)
  * WordPress and PHP development
  * Basic JavaScript
  
SKILLS: JavaScript, React, Node.js, Express, Python basics, MongoDB, PostgreSQL, AWS, Azure
EDUCATION: Bootcamp Graduate

Not ideal - primarily JavaScript/Node.js background, Python as secondary skill.
            """
        ),
        CV(
            id="cv_004",
            candidate_name="David Brown",
            raw_text="""
David Brown
Principal Software Engineer

EXPERIENCE:
- Principal Engineer at EnterpriseCorps (5 years)
  * Architect of distributed systems for 50M+ users
  * Expert in system design and scaling
  * Strong Python and C++ background
  * Led teams of 8+ engineers
  * Cloud architect for multi-region deployments
  
- Senior Software Engineer at MegaTech (4 years)
  * Built high-throughput systems
  * Database optimization expert
  
SKILLS: Python, C++, Java, System Architecture, Cloud (AWS, GCP), PostgreSQL, MongoDB, Leadership
EDUCATION: M.S. Computer Science

Overqualified but excellent system design and cloud experience.
            """
        ),
        CV(
            id="cv_005",
            candidate_name="Emma Davis",
            raw_text="""
Emma Davis
Frontend Developer

EXPERIENCE:
- Frontend Developer at UIDesignCo (3 years)
  * React and Vue.js expert
  * CSS and design systems
  * HTML5 and responsive design
  * Some TypeScript experience
  
- Junior Frontend Developer at WebAgency (1.5 years)
  * jQuery and vanilla JavaScript
  * WordPress theme development
  
SKILLS: JavaScript, React, Vue.js, HTML, CSS, TypeScript basics, Git
EDUCATION: Diploma in Web Design

Not suitable - specialized in frontend, minimal Python or backend experience.
            """
        ),
    ]
    
    return cvs


def get_sample_jd() -> JobDescription:
    """Get sample job description.
    
    Returns a Senior Python Backend Engineer job description with specific
    requirements for FastAPI, PostgreSQL, and cloud experience.
    
    Returns:
        JobDescription object.
    """
    jd = JobDescription(
        role="Senior Python Backend Engineer",
        requirements="""
POSITION: Senior Python Backend Engineer

We are seeking an experienced Senior Python Backend Engineer to join our growing engineering team.

REQUIREMENTS:
- 4+ years of professional Python backend development experience
- Strong expertise with FastAPI or similar modern Python web frameworks
- Solid experience with PostgreSQL database design and optimization
- Demonstrated experience with cloud platforms (AWS preferred, but GCP/Azure acceptable)
- Excellent system design and architectural thinking skills
- Experience with Docker and containerization
- Strong understanding of REST API design principles
- Proficiency with Git and modern development workflows
- Experience with microservices architecture
- Ability to mentor junior developers

PREFERRED:
- Kubernetes experience
- CI/CD pipeline implementation
- Experience with message queues (RabbitMQ, Kafka)
- Familiarity with distributed systems
- Open source contributions

RESPONSIBILITIES:
- Design and implement scalable backend services
- Optimize database performance and queries
- Collaborate with cross-functional teams
- Mentor and guide junior engineers
- Implement and maintain CI/CD pipelines
- Participate in architectural discussions and design reviews

We offer competitive compensation, remote flexibility, and opportunities for growth.
        """
    )
    
    return jd
