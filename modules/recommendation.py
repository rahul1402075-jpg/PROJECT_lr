"""
recommendation.py
-------------------
MODULE 14: RECOMMENDATION ENGINE

A simple, easy-to-explain RULE-BASED recommendation engine.

Why rule-based (for viva)?
---------------------------
A rule-based system is transparent and easy to justify to an examiner:
"If the predicted interest is Technology, we recommend these five
specific topics." This keeps the project's scope reasonable for a
college mini-project while the ML model handles the harder task of
classification. A rule-based lookup table is simple to extend later
with collaborative filtering or content-based filtering algorithms
(see Future Enhancements in the README).
"""

RECOMMENDATIONS = {
    "Technology": ["Artificial Intelligence", "Python Programming", "Machine Learning",
                   "Cloud Computing", "Cybersecurity"],
    "Sports": ["Cricket", "Football", "Sports News", "Match Updates", "Fitness Training"],
    "Entertainment": ["Movies", "Web Series", "Music", "Entertainment News", "Celebrity Interviews"],
    "Politics": ["Election Updates", "Policy Analysis", "Government News", "Public Debates"],
    "Business": ["Startup News", "Stock Market Updates", "Entrepreneurship", "Business Strategy"],
    "Education": ["Online Courses", "Programming Tutorials", "Study Resources", "Certifications"],
    "Health": ["Fitness Tips", "Mental Wellness", "Nutrition Guides", "Yoga & Meditation"],
    "Travel": ["Travel Guides", "Budget Trip Planning", "Adventure Destinations", "Travel Photography"],
    "Gaming": ["New Game Releases", "Esports Tournaments", "Game Reviews", "Streaming Setups"],
    "Finance": ["Investing Basics", "Personal Budgeting", "Cryptocurrency News", "Financial Planning"],
}


def get_recommendations(category):
    """
    Returns a list of recommended topics for a given predicted interest
    category. Returns a generic message if the category is unknown so
    the app never crashes on unexpected input.
    """
    return RECOMMENDATIONS.get(category, ["General News", "Trending Topics"])
