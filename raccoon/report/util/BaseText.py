from dataclasses import dataclass


@dataclass
class BaseText:
    intro_text: str
    metric_guide_text: str    
    metric_text: str    
    practical_interpretation_text: str
    throughput_text: str
    throughput_interpretation_text: str
    rank_distribution_text: str
    rank_distribution_interpretation_text: str
    def __post_init__(self):
        self.intro_text = self.intro_text
        self.metric_guide_text = self.metric_guide_text
        self.metric_text = self.metric_text
        self.practical_interpretation_text = self.practical_interpretation_text
        self.throughput_text = self.throughput_text
        self.throughput_interpretation_text = self.throughput_interpretation_text
        self.rank_distribution_text = self.rank_distribution_text
        self.rank_distribution_interpretation_text = self.rank_distribution_interpretation_text