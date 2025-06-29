FAMILY_PERCENTAGES = {
    'CurveFault': 0.180559,
    'FlatFault':  0.271263,
    'CurveVel':   0.183491,
    'FlatVel':    0.180483,
    'Style':      0.184205,
}

FAMILY_MAE_REFS = {
    'CurveFault': 740.6580,
    'FlatFault':  766.6056,
    'CurveVel':   713.8649,
    'FlatVel':    724.1204,
    'Style':      558.5109,
}

def calculate_estimated_mae(family_name: str, mae_general: float, mae_experimental: float) -> float:
    if family_name not in FAMILY_PERCENTAGES:
        raise ValueError(f"Error: El nombre de la familia '{family_name}' no es válido.")
    p_family = FAMILY_PERCENTAGES[family_name]
    mae_ref_family = FAMILY_MAE_REFS[family_name]
    estimated_mae = ((mae_general - mae_experimental) / p_family) + mae_ref_family
    
    return estimated_mae

if __name__ == '__main__':
    print("--- Calculadora de MAE estimado por familia ---")
    mae_general = 306.5 
    family_name = 'Style'
    mae_experimental = 396.1

    try:
        family_mae = calculate_estimated_mae(
            family_name=family_name,
            mae_general=mae_general,
            mae_experimental=mae_experimental
        )
        
        print("\n--- RESULTADO ---")
        print(f"Para un modelo con MAE general de {mae_general}:")
        print(f"El MAE estimado para la familia '{family_name}' es: {family_mae:.2f} m/s")

    except ValueError as e:
        print(e)