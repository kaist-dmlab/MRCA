class GenerationConfig:
    def __init__(self, total_gen=1203):
        """
        Initialize the GenerationConfig class.

        Parameters:
        - total_gen (int): Total number of generations to be created.
        """
        self.total_gen = total_gen
        self.config_list = []

    def add_config(self, prompt="", cfg=5.0, fg_criterion='entropy', fg_scale=0.03, cls_index=None, inst_index = None):
        """
        Add a new generation configuration to the list.

        Parameters:
        - prompt (str): Text prompt for generation.
        - cfg (float): Classifier-Free Guidance scale.
        - fg_criterion (str): Criterion for selecting foreground objects.
        - fg_scale (float): Scale for foreground objects.
        - cls_index (int or None): Class index for generation.
        """
        config = {
            "prompt": prompt,
            "cfg": cfg,
            "fg_criterion": fg_criterion,
            "fg_scale": fg_scale,
            "cls_index": cls_index,
            "inst_index": inst_index,
        }
        self.config_list.append(config)

    

    def get_config(self, index):
        """
        Retrieve a specific configuration by index.

        Parameters:
        - index (int): Index of the configuration to retrieve.

        Returns:
        - dict: The requested configuration.
        """
        if 0 <= index < len(self.config_list):
            return self.config_list[index]
        else:
            raise IndexError("Configuration index out of range.")

    def reset(self):
        self.config_list = []