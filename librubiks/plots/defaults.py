import matplotlib.pyplot as plt
import matplotlib.colors as mcolour

rc_params = { "font.size": 24, "legend.fontsize": 22, "legend.framealpha": 0.5 }  # matplotlib settings
rc_params_small = { **rc_params, "font.size": 20, "legend.fontsize": 18 }  # Same but with smaller font

base_colours = list(mcolour.BASE_COLORS)
tab_colours = list(mcolour.TABLEAU_COLORS)
all_colours = base_colours[:-1] + tab_colours[:-2]  # 15 unique matplotlib colours

