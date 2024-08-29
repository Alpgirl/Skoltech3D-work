def get_elevation_plates(object_h, hard_base_h=.5, desired_center_h=36):
    r"""Calculates the heights of elevation plates
    needed to raise the center of an object with the given height to the desired height."""
    elevation = round(desired_center_h - hard_base_h - object_h / 2)
    bits = list(reversed(f'{elevation:b}'))
    return [2 ** i for i in range(len(bits)) if bits[i] == '1']
