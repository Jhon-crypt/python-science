import numpy as np
import pandas as pd
from pyDOE3 import *
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm

class DOEAnalysis:
    def __init__(self, factor_ranges):
        """
        Initialize DOE analysis with factor ranges
        factor_ranges: dict with format {'factor_name': (min_value, max_value)}
        """
        self.factor_ranges = factor_ranges
        self.factors = len(factor_ranges)
        self.factor_names = list(factor_ranges.keys())

    def scale_to_real(self, coded_value, range_min, range_max):
        """Convert coded values (-1 to +1) to real values"""
        return (range_max + range_min)/2 + ((range_max - range_min)/2)*coded_value

    def create_ccd(self, center_points=(4,4), alpha='rotatable', face='ccc'):
        """Create Central Composite Design"""
        # Generate CCD design
        ccd = ccdesign(self.factors, center=center_points, alpha=alpha, face=face)
        
        # Scale to real values
        real_values = np.zeros_like(ccd)
        for i in range(self.factors):
            min_val = list(self.factor_ranges.values())[i][0]
            max_val = list(self.factor_ranges.values())[i][1]
            real_values[:,i] = self.scale_to_real(ccd[:,i], min_val, max_val)
        
        # Create DataFrame
        self.ccd_df = pd.DataFrame(real_values, columns=self.factor_names)
        return self.ccd_df

    def create_bbdesign(self, center_points=3):
        """Create Box-Behnken Design"""
        if self.factors < 3:
            raise ValueError("Box-Behnken design requires at least 3 factors")
        
        # Generate BBD design
        bbd = bbdesign(self.factors, center=center_points)
        
        # Scale to real values
        real_values = np.zeros_like(bbd)
        for i in range(self.factors):
            min_val = list(self.factor_ranges.values())[i][0]
            max_val = list(self.factor_ranges.values())[i][1]
            real_values[:,i] = self.scale_to_real(bbd[:,i], min_val, max_val)
        
        # Create DataFrame
        self.bbd_df = pd.DataFrame(real_values, columns=self.factor_names)
        return self.bbd_df

    def add_response(self, design_df, response_values, response_name='Yield'):
        """Add experimental results to the design"""
        design_df[response_name] = response_values
        return design_df

    def plot_design_points(self, design_df, design_type="CCD"):
        """Create 3D visualization of design points"""
        if self.factors != 3:
            raise ValueError("3D plot only works with 3 factors")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(design_df[self.factor_names[0]], 
                  design_df[self.factor_names[1]], 
                  design_df[self.factor_names[2]], 
                  c='blue', marker='o')
        
        ax.set_xlabel(self.factor_names[0])
        ax.set_ylabel(self.factor_names[1])
        ax.set_zlabel(self.factor_names[2])
        plt.title(f"{design_type} Design Points")
        plt.show()

def main():
    # Example usage
    # Define factor ranges for a chemical process
    factor_ranges = {
        'Temperature': (60, 80),  # °C
        'Pressure': (1, 5),      # bar
        'Concentration': (0.1, 0.5) # mol/L
    }

    # Initialize DOE analysis
    doe = DOEAnalysis(factor_ranges)

    # Create CCD design
    print("\nCentral Composite Design:")
    ccd_df = doe.create_ccd()
    print(ccd_df)
    doe.plot_design_points(ccd_df, "Central Composite")

    # Create BBD design
    print("\nBox-Behnken Design:")
    bbd_df = doe.create_bbdesign()
    print(bbd_df)
    doe.plot_design_points(bbd_df, "Box-Behnken")

    # Example: Adding experimental results (simulated data)
    # In real case, you would replace this with actual experimental results
    np.random.seed(42)
    simulated_responses = 75 + np.random.normal(0, 5, len(ccd_df))
    ccd_df_with_response = doe.add_response(ccd_df, simulated_responses)
    
    print("\nDesign with responses:")
    print(ccd_df_with_response)

if __name__ == "__main__":
    main()
