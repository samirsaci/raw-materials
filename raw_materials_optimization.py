"""
Raw Materials Optimization for Food Manufacturing

This script uses linear programming to create an optimal recipe
that meets nutritional requirements at minimum cost.
"""

import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpStatus, value


def load_data():
    """Load nutrition and cost data from data folder."""
    nutrition = pd.read_excel("data/Nutrition Facts.xlsx", index_col=0)
    costs = pd.read_excel("data/Costs.xlsx")
    dict_costs = dict(zip(costs["Ingredients"], costs["Costs"]))
    return nutrition, dict_costs


def create_sample_data():
    """Create sample data if files not available."""
    nutrition = pd.DataFrame({
        "Protein": [0.10, 0.20, 0.15, 0.00, 0.04, 0.033, 0.258],
        "Fat": [0.08, 0.10, 0.11, 0.01, 0.01, 0.013, 0.492],
        "Fibre": [0.001, 0.005, 0.003, 0.10, 0.15, 0.028, 0.085],
        "Salt": [0.002, 0.005, 0.007, 0.002, 0.008, 0.000, 0.001],
        "Sugar": [0.000, 0.000, 0.000, 0.000, 0.000, 0.045, 0.047],
    }, index=["Chicken", "Beef", "Mutton", "Rice", "Wheat bran", "Corn", "Peanuts"])

    dict_costs = {
        "Chicken": 0.095,
        "Beef": 0.150,
        "Mutton": 0.100,
        "Rice": 0.002,
        "Wheat bran": 0.005,
        "Corn": 0.012,
        "Peanuts": 0.013,
    }

    return nutrition, dict_costs


def optimize_recipe(nutrition, dict_costs, bar_weight=100, constraints=None):
    """
    Optimize meal bar recipe to minimize cost.

    Default nutritional constraints (per 100g bar):
    - Protein: >= 22g
    - Fat: <= 22g
    - Fibre: >= 6g
    - Salt: <= 3g
    - Sugar: <= 20g
    """
    if constraints is None:
        constraints = {
            "Protein": (">=", 22),
            "Fat": ("<=", 22),
            "Fibre": (">=", 6),
            "Salt": ("<=", 3),
            "Sugar": ("<=", 20),
        }

    ingredients = list(nutrition.index)

    # Initialize model
    model = LpProblem("Meal_Bar_Recipe", LpMinimize)

    # Decision variables: grams of each ingredient
    x = LpVariable.dicts("qty", ingredients, lowBound=0, cat="continuous")

    # Objective: minimize cost
    model += lpSum([dict_costs[i] * x[i] for i in ingredients])

    # Constraint: total weight = bar_weight
    model += lpSum([x[i] for i in ingredients]) == bar_weight

    # Nutritional constraints
    for nutrient, (op, limit) in constraints.items():
        if nutrient in nutrition.columns:
            nutrient_sum = lpSum(
                [x[i] * nutrition.loc[i, nutrient] for i in ingredients]
            )
            if op == ">=":
                model += nutrient_sum >= limit
            else:
                model += nutrient_sum <= limit

    # Solve
    model.solve()

    return model, x, ingredients


def display_results(model, x, ingredients, nutrition, dict_costs, bar_weight=100):
    """Display optimization results."""
    print("=" * 60)
    print("MEAL BAR RECIPE OPTIMIZATION")
    print("=" * 60)

    print(f"\nStatus: {LpStatus[model.status]}")

    if model.status != 1:  # Not optimal
        print("No feasible solution found. Try relaxing constraints.")
        return

    print(f"Cost per Bar ({bar_weight}g): ${value(model.objective):.2f}")

    print("\n" + "-" * 60)
    print("OPTIMAL RECIPE")
    print("-" * 60)
    print(f"{'Ingredient':<15} {'Quantity (g)':<15} {'Cost ($)':<10}")
    print("-" * 40)

    total_weight = 0
    for ingredient in ingredients:
        qty = x[ingredient].varValue
        if qty and qty > 0.01:
            cost = qty * dict_costs[ingredient]
            print(f"{ingredient:<15} {qty:<15.2f} {cost:<10.4f}")
            total_weight += qty

    print("-" * 40)
    print(f"{'Total':<15} {total_weight:<15.2f} {value(model.objective):<10.4f}")

    # Nutritional profile
    print("\n" + "-" * 60)
    print("NUTRITIONAL PROFILE")
    print("-" * 60)

    for nutrient in nutrition.columns:
        total = sum(
            (x[i].varValue or 0) * nutrition.loc[i, nutrient]
            for i in ingredients
        )
        print(f"{nutrient}: {total:.2f}g")


def sensitivity_analysis(nutrition, dict_costs):
    """Analyze how cost changes with different protein requirements."""
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: Protein Requirement")
    print("=" * 60)

    protein_levels = [15, 18, 20, 22, 25, 28, 30]
    print(f"{'Protein (g)':<15} {'Cost ($)':<15} {'Status':<15}")
    print("-" * 45)

    for protein in protein_levels:
        constraints = {
            "Protein": (">=", protein),
            "Fat": ("<=", 25),
            "Fibre": (">=", 5),
            "Salt": ("<=", 3),
            "Sugar": ("<=", 20),
        }
        model, x, ingredients = optimize_recipe(nutrition, dict_costs, 100, constraints)
        status = LpStatus[model.status]
        cost = value(model.objective) if model.status == 1 else float('inf')
        print(f"{protein:<15} ${cost:<14.2f} {status:<15}")


def main():
    """Main function."""
    try:
        nutrition, dict_costs = load_data()
        print("Data loaded from Excel files.")
    except FileNotFoundError:
        print("Excel files not found. Using sample data.")
        nutrition, dict_costs = create_sample_data()

    # Display input data
    print("\n--- NUTRITION FACTS (per gram) ---")
    print(nutrition)

    print("\n--- INGREDIENT COSTS ($/gram) ---")
    for ing, cost in dict_costs.items():
        print(f"  {ing}: ${cost:.3f}")

    # Optimize with default constraints
    model, x, ingredients = optimize_recipe(nutrition, dict_costs)
    display_results(model, x, ingredients, nutrition, dict_costs)

    # Sensitivity analysis
    sensitivity_analysis(nutrition, dict_costs)


if __name__ == "__main__":
    main()
