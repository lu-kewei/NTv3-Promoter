import logging 
from accelerate.logging import MultiProcessAdapter 

def Logger(logfile,name=None,only_message=False,show=True,accelerate=True):
    logger = logging.getLogger(name)
 

    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if only_message:
        formatter = logging.Formatter("%(message)s")
    else:
        formatter = logging.Formatter("%(asctime)s - %(filename)s [line: %(lineno)s] - %(message)s")
    fh = logging.FileHandler(logfile,mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    if show:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.addHandler(fh)
    if accelerate:
        return MultiProcessAdapter(logger, {})
    else:
        return logger
