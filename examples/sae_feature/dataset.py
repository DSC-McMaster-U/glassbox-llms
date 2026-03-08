"""
Shared dataset utilities for SAE feature analysis examples.

Provides diverse text datasets from Hugging Face for training SAEs
and analyzing their learned features.
"""


def create_diverse_dataset():
    """
    Load diverse texts from Hugging Face datasets.
    
    Loads texts from multiple domains:
    - News articles (ag_news)
    - Movie reviews (imdb)
    - Scientific papers (arxiv abstracts)
    - Wikipedia articles
    
    Falls back to hardcoded dataset if HF datasets fail to load.
    
    Returns:
        List[str]: List of text strings for training.
    """
    from datasets import load_dataset
    
    texts = []
    
    print("    Loading diverse datasets from Hugging Face...")
    
    # 1. News articles (200 texts)
    try:
        news = load_dataset("ag_news", split="train[:200]", trust_remote_code=True)
        texts.extend([item["text"] for item in news])
        print(f"      ✓ Loaded {len(news)} news articles")
    except Exception as e:
        print(f"      ✗ Could not load ag_news: {e}")
    
    # 2. Movie reviews (200 texts, truncated)
    try:
        reviews = load_dataset("imdb", split="train[:200]", trust_remote_code=True)
        texts.extend([item["text"][:400] for item in reviews])
        print(f"      ✓ Loaded {len(reviews)} movie reviews")
    except Exception as e:
        print(f"      ✗ Could not load imdb: {e}")
    
    # 3. Scientific abstracts (200 texts)
    try:
        science = load_dataset("scientific_papers", "arxiv", split="train[:200]", trust_remote_code=True)
        texts.extend([item["abstract"][:400] for item in science])
        print(f"      ✓ Loaded {len(science)} scientific abstracts")
    except Exception as e:
        print(f"      ✗ Could not load scientific_papers: {e}")
    
    # 4. Wikipedia articles (200 texts, truncated)
    try:
        wiki = load_dataset("wikipedia", "20220301.en", split="train[:200]", trust_remote_code=True)
        texts.extend([item["text"][:400] for item in wiki])
        print(f"      ✓ Loaded {len(wiki)} Wikipedia excerpts")
    except Exception as e:
        print(f"      ✗ Could not load wikipedia: {e}")
    
    # Fallback if datasets fail to load
    if len(texts) < 100:
        print("      ⚠ Using hardcoded fallback dataset")
        texts = _create_hardcoded_fallback()
    
    print(f"    Total texts: {len(texts)}")
    return texts


def _create_hardcoded_fallback():
    """
    Hardcoded fallback dataset in case Hugging Face datasets fail.
    
    Returns:
        List[str]: List of diverse text examples across multiple domains.
    """
    texts = [
        # Programming
        "Python programming language is popular.",
        "She writes code in Python daily.",
        "The Python interpreter executes programs.",
        "import torch  # PyTorch deep learning",
        "JavaScript runs in web browsers.",
        "The compiler optimizes the code.",
        "Debugging helps find software bugs.",
        "Version control tracks code changes.",
        
        # Gold theme
        "The golden sunset was beautiful.",
        "He won a gold medal yesterday.",
        "Silence is golden in the library.",
        "Golden retrievers are loyal dogs.",
        "The gold ring sparkled brightly.",
        "They struck gold in California.",
        "Golden opportunity knocked once.",
        "Gold prices fluctuate daily.",
        
        # Emotions
        "I love this amazing movie!",
        "Terrible service, very disappointed.",
        "The happiest day of my life!",
        "Awful experience, never again.",
        "Joy filled the celebration room.",
        "Sadness overwhelmed her completely.",
        "Anger rarely solves problems.",
        "Fear can paralyze decision-making.",
        
        # Science
        "The Mars rover explores terrain.",
        "Einstein revolutionized physics completely.",
        "DNA encodes genetic information.",
        "Quantum mechanics explains atoms.",
        "Photosynthesis converts sunlight energy.",
        "Neurons transmit electrical signals.",
        "Evolution shaped diverse species.",
        "Gravity pulls objects together.",
        
        # Food
        "The chef prepared delicious pasta.",
        "Fresh bread smells wonderful.",
        "Chocolate cake tastes sweet.",
        "Spicy curry burns pleasantly.",
        "Coffee energizes morning routines.",
        "Sushi combines fish rice.",
        "Pizza delivers cheesy satisfaction.",
        "Salad provides healthy nutrients.",
        
        # Sports
        "Basketball game ended overtime.",
        "Marathon runners train daily.",
        "Soccer match was exciting.",
        "Swimming builds strong muscles.",
        "Tennis requires quick reflexes.",
        "Cycling improves cardiovascular health.",
        "Baseball fans love statistics.",
        "Golf demands mental focus.",
        
        # Nature
        "The forest has many trees.",
        "Birds migrate south annually.",
        "Ocean waves crash loudly.",
        "Mountains reach high peaks.",
        "Rivers flow towards oceans.",
        "Deserts receive minimal rainfall.",
        "Rainforests teem with life.",
        "Glaciers slowly reshape landscapes.",
        
        # Technology
        "Artificial intelligence learns patterns.",
        "The algorithm optimizes performance.",
        "Cloud computing stores data.",
        "Social media connects people.",
        "Smartphones revolutionized communication.",
        "Blockchain enables secure transactions.",
        "Robotics automates manufacturing processes.",
        "Quantum computers solve problems.",
        
        # History
        "Ancient Rome conquered territories.",
        "World War changed everything.",
        "Renaissance sparked artistic innovation.",
        "Industrial revolution transformed society.",
        "Democracy emerged in Greece.",
        "Explorers discovered new continents.",
        "Revolutions toppled monarchies.",
        "Civilizations rose and fell.",
        
        # Medicine
        "Vaccines prevent deadly diseases.",
        "Surgery repairs damaged organs.",
        "Antibiotics fight bacterial infections.",
        "Exercise promotes cardiovascular health.",
        "Therapy helps mental wellness.",
        "Nutrition affects overall health.",
        "Sleep restores body functions.",
        "Meditation reduces stress levels.",
        
        # Business
        "Markets determine product prices.",
        "Competition drives innovation forward.",
        "Investment generates financial returns.",
        "Marketing influences consumer behavior.",
        "Supply chains distribute goods.",
        "Entrepreneurs create new ventures.",
        "Economy cycles through phases.",
        "Trade connects global markets.",
        
        # Education
        "Teachers inspire student learning.",
        "Research expands human knowledge.",
        "Universities offer advanced degrees.",
        "Libraries provide information access.",
        "Curiosity drives intellectual growth.",
        "Practice develops professional skills.",
        "Mentors guide career development.",
        "Learning never truly stops.",
    ]
    return texts
