from dataclasses import dataclass

@dataclass
class Patient:
    gender: int  # 1 = male, 2 = female
    weight_kg: float
    height_cm: float
    age: int
    hematocrit: float  # as a fraction, e.g., 0.45
    serum_creatinine: float  # in mg/dL

    def compute_bsa(self):
        """Calculate body surface area using the DuBois & DuBois formula."""
        return 0.007184 * (self.height_cm ** 0.725) * (self.weight_kg ** 0.425)

    def compute_blood_volume(self):
        """Estimate total blood volume based on gender and weight."""
        return self.weight_kg * (72 if self.gender == 1 else 65)  # mL

    def compute_plasma_volume(self):
        """Calculate plasma volume from blood volume and hematocrit."""
        return self.compute_blood_volume() * (1 - self.hematocrit)  # mL

    def compute_circulatory_volumes(self):
        """Break plasma volume into CS, RCS, and HCS compartments."""
        Vplasma = self.compute_plasma_volume()
        VRCS = 0.02 * Vplasma
        VHCS = 0.10 * Vplasma
        VCS = Vplasma - VRCS - VHCS
        return Vplasma, VCS, VRCS, VHCS

    def compute_ecf_icf(self):
        """
        Estimate TBW, ECW, ICW, and ISF using BMI-dependent fractions from Ritz et al. (2008),
        which classify patients as lean, overweight, or obese based on BMI.
        """
        bmi = self.weight_kg / ((self.height_cm / 100) ** 2)

        # Determine BMI category
        if bmi < 25:
            bmi_category = 'lean'
        elif bmi < 30:
            bmi_category = 'overweight'
        else:
            bmi_category = 'obese'

        # Assign TBW/FFM (%) and ECW/TBW (%) from Ritz et al. (2008)
        if self.gender == 1:  # male
            TBW_FFM_fraction = {'lean': 0.74, 'overweight': 0.753, 'obese': 0.768}[bmi_category]
            ECW_TBW_fraction = {'lean': 0.403, 'overweight': 0.418, 'obese': 0.435}[bmi_category]
        else:  # female
            TBW_FFM_fraction = {'lean': 0.747, 'overweight': 0.743, 'obese': 0.733}[bmi_category]
            ECW_TBW_fraction = {'lean': 0.379, 'overweight': 0.409, 'obese': 0.437}[bmi_category]

        # Estimate fat free mass (FFM) using simple sex-based estimate (can be replaced with better model if available)
        FFM = self.weight_kg * (0.85 if self.gender == 1 else 0.75)

        # Calculate TBW, ECW, ICW
        TBW = FFM * TBW_FFM_fraction  # liters
        ECW = TBW * ECW_TBW_fraction  # liters
        ICW = TBW - ECW               # liters

        # Calculate ISF: ISF = ECW - plasma volume
        plasma_vol_L = self.compute_plasma_volume() / 1000  # convert mL to L
        ISF = ECW - plasma_vol_L

        return TBW, ECW, ICW, ISF

    def compute_gfr_ckd_epi(self, race="asian"):
        """
        Estimate glomerular filtration rate (eGFR) using the four-level CKD-EPI equation
        from Stevens et al. (Kidney Int 2011), supporting Black, Asian, Native American/Hispanic,
        and White/Other groups. Units: mL/min/1.73 m².
        """
        scr = self.serum_creatinine 
        kappa = 0.7 if self.gender == 2 else 0.9
        alpha = -0.328 if self.gender == 2 else -0.412
        min_scr_k = min(scr / kappa, 1)
        max_scr_k = max(scr / kappa, 1)
        gender_factor = 1.018 if self.gender == 2 else 1.0
        
        race = race.lower()
        if race == "black":
            race_factor = 1.16
        elif race == "asian":
            race_factor = 1.05
        elif race in ("native_american", "hispanic", "native_american_hispanic"):
            race_factor = 1.01
        elif race in ("white", "other", "white_other"):
            race_factor = 1.0
        else:
            raise ValueError("Invalid race. Choose from 'black', 'asian', 'native_american_hispanic', or 'white_other'.")
    
        return 141 * (min_scr_k ** alpha) * (max_scr_k ** -1.209) * (0.993 ** self.age) * gender_factor * race_factor
    
@dataclass
class PatientExtended(Patient):
    def compute_cardiac_output(self):
        """L/h: CO = 159 × BSA − 1.56 × Age + 114"""
        bsa = self.compute_bsa()
        return 159 * bsa - 1.56 * self.age + 114

    def compute_renal_plasma_flow(self):
        """Renal % CO = −8.7 × BSA + 0.29 × Height − 0.081 × Age − 13"""
        bsa = self.compute_bsa()
        renal_fraction = -8.7 * bsa + 0.29 * self.height_cm - 0.081 * self.age - 13
        co = self.compute_cardiac_output()
        return (renal_fraction / 100) * co

    def compute_hepatic_plasma_flow(self):
        """Hepatic % CO = −0.108 x Age + 1.04 × Sex + 27.9"""
        hepatic_fraction = -0.108 * self.age + 1.04 * self.gender + 27.9
        co = self.compute_cardiac_output()
        return (hepatic_fraction / 100) * co

# Create patient
p = PatientExtended(gender=2, weight_kg=65, height_cm=160, age=61.36618754, hematocrit=0.412, serum_creatinine=62.1*0.0113)

# Get physiological data
#print("BSA:", p.compute_bsa())
#print("Plasma Volume:", p.compute_plasma_volume())
#print("Circulatory Volumes:", p.compute_circulatory_volumes())
#print("TBW, ECW, ICW, ISF:", p.compute_ecf_icf())
#print("GFR (CKD-EPI):", p.compute_gfr_ckd_epi())

# Compute flow rates
#print("Cardiac Output (L/h):", p.compute_cardiac_output())
#print("Renal Plasma Flow (L/h):", p.compute_renal_plasma_flow())
#print("Hepatic Plasma Flow (L/h):", p.compute_hepatic_plasma_flow())