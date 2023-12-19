import gin

@gin.configurable
class XLAFp8Config:
  def __init__(self, enabled=False):
    self.enabled = enabled

  def __str__(self):
    return f"XLAFp8Config: enabled:{self.enabled}."
