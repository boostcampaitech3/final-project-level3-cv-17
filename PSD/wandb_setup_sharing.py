import wandb

api_key = ''
project_name = ''


def wandb_login():
    wandb.login(key=api_key)

def wandb_init(opt, work_dir_exp):
    wandb.init(
        project=project_name,
        entity="mg_generation",
        name=work_dir_exp.split('/')[-1],
        reinit=True,
        config=opt.__dict__,
    )