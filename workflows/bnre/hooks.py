def get_tune_gamma(epochs, gamma):
    increase = gamma/epochs * 1.25
    max_value = gamma
    def tune_gamma(trainer, **kwargs):
        trainer._criterion.gamma += increase
        if trainer._criterion.gamma >= max_value:
            trainer._criterion.gamma = max_value

    return tune_gamma

def reset_gamma(trainer, **kwargs):
    trainer._criterion.gamma = 0.

def load_hooks(arguments, trainer):
    trainer.add_event_handler(trainer.events.epoch_complete, get_tune_gamma(arguments.epochs, arguments.gamma))
    trainer.add_event_handler(trainer.events.fit_start, reset_gamma)

