import json
import os
import numpy as np

class SimulationProperties():
    """ Used to load simulation properties form JSON files.
    """

    def __init__(self, file=""):
        # Default parameter
        self.dt = None
        self.tf = None
        self.g = None
        self.k_p = None
        self.J0 = None
        self.J0_factor = None
        self.m = None
        self.m_mmu = None
        self.mmu_min = None
        self.mmu_max = None
        self.r_true = None
        self.r_guess = None
        self.attitude_initial = None
        self.angular_vel_initial = None
        self.v_mmu = None
        self.file = file

        # Read in json file
        self.json = self.readJSON()

        # Apply settings
        self.setSettings()

        
    def readJSON(self, filepath="simulation_properties"):
        """ Reads in a JSON file and returns JSON-Python object.
            JSONs have to be in simulation_properties folder.
        """

        ## Checking if JSON property file was specified
        if self.file == "":
            raise ValueError("No JSON config file specified!")
        
        ## Checking if JSON files are present at folderPath
        files = [file for file in os.listdir(filepath) if file.endswith('.json')]

        if len(files) == 0:
            raise FileNotFoundError("Did not find any JSON simulation property files!")
        
        ## Determining if file exists
        found = False
        for file in files:
            if file == self.file:
                found = True

        if not found:
            raise FileNotFoundError(f"Could not find file {self.file}")
        
        settings = ""
        with open(os.path.join(filepath, self.file), 'r') as file:
            settings = json.load(file)

        return settings
    
    def setSettings(self):
        """ Applies the settings from the JSON to the class parameters.
        """

        if self.json == dict():
            raise AttributeError("Settings could not be found!")
        
        ## Simulation Settings
        self.dt = self.json['simulation_settings']['dt']
        self.tf = self.json['simulation_settings']['tf']

        ## Simulator Settings
        self.g = self.json['simulator_settings']['g']

        # Parsing of optional k_p field
        if 'simulator_settings' in self.json and 'k_p' in self.json['simulator_settings']:
            self.k_p = float(self.json['simulator_settings']['k_p'])

        self.J0 = np.array(self.json['simulator_settings']['J0'])

        # Parsing of optional J0_factor field
        if 'simulator_settings' in self.json and 'J0_factor' in self.json['simulator_settings']:
            self.J0 *= self.json['simulator_settings']['J0_factor']

        self.m = self.json['simulator_settings']['m']

        # Parsing of optional m_mmu field
        if 'simulator_settings' in self.json and 'm_mmu' in self.json['simulator_settings']:
            self.m_mmu = float(self.json['simulator_settings']['m_mmu'])
        if 'simulator_settings' in self.json and 'mmu_min' in self.json['simulator_settings']:
            self.mmu_min = float(self.json['simulator_settings']['mmu_min'])
        if 'simulator_settings' in self.json and 'mmu_max' in self.json['simulator_settings']:
            self.mmu_max = float(self.json['simulator_settings']['mmu_max'])

        self.r_true = np.array(self.json['simulator_settings']['r_true']).flatten()
        self.attitude_initial = np.array(self.json['simulator_settings']['attitude_initial']).flatten()
        self.angular_vel_initial = np.array(self.json['simulator_settings']['angular_vel_initial']).flatten()

        # Parsing of optional r_guess parameter
        if 'simulator_settings' in self.json and 'r_guess' in self.json['simulator_settings']:
            self.r_guess = np.array(self.json['simulator_settings']['r_guess']).flatten()
        else:
            self.r_guess = self.r_true

        # Parsing of optional v_mmu field
        if 'simulator_settings' in self.json and 'v_mmu' in self.json['simulator_settings']:
            self.v_mmu = float(self.json['simulator_settings']['v_mmu'])
    
    def toStr(self) -> str:        
        return f"Simulation Settings:\n" \
               f"  dt: {self.dt}\n" \
               f"  tf: {self.tf}\n" \
               f"Simulator Settings:\n" \
               f"  g: {self.g}\n" \
               f"  J0: {self.matrix_to_string(self.J0)}\n" \
               f"  m: {self.m}\n" \
               f"  r_true: {self.r_true}\n" \
               f"  r_guess: {self.r_guess}\n" \
               f"  attit_init: {self.array_to_string(self.attitude_initial)}\n" \
               f"  ang_vel_init: {self.array_to_string(self.angular_vel_initial)}"\
               f"  k_p: {self.k_p}\n"\
               f"  m_mmu: {self.m_mmu}\n"\
               f"  mmu_min: {self.mmu_min}\n"\
               f"  mmu_max: {self.mmu_max}\n"\
               f"  v_mmu: {self.v_mmu}"

    def matrix_to_string(self,matrix):
            return '[' + ' '.join([str(row) for row in matrix]) + ']'

    def array_to_string(self, array):
        return '[' + ' '.join([str(element) for element in array]) + ']'
