import argparse
from patchcore_runner import PatchCore
from data.mvtec3d import mvtec3d_classes
from data.real3d import real3d_classes
from data.anomalyshape import shapenet3d_classes
from data.MulSen import mulsen_classes
from data.MiniShiftAD import minishiftAD_classes
# from data.quan import quan_classes  # 注释掉：文件不存在
import pandas as pd
import torchvision
import os
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path('logs')


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    if value.lower() in ('false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError('Expected True or False.')


class TeeLogger:
    """同时输出到终端和日志文件"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def setup_logging():
    """设置日志文件，默认保存到 autodl-tmp/log/ 或本地 logs/"""
    global LOG_DIR
    # 尝试 AutoDL 路径（Linux），失败则用本地路径
    if os.name == 'posix':  # Linux/Unix
        log_dirs = [
            Path("/root/autodl-tmp/log"),
            Path("autodl-tmp/log"),
        ]
    else:  # Windows
        log_dirs = [
            Path("logs"),
        ]

    log_dir = None
    for d in log_dirs:
        try:
            d.mkdir(parents=True, exist_ok=True)
            log_dir = d
            break
        except Exception as e:
            continue

    if log_dir is None:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR = log_dir

    # 生成日志文件名：2026-10-26-20-19.txt
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_file = log_dir / f"{timestamp}.txt"

    # 重定向 stdout 和 stderr
    tee_stdout = TeeLogger(log_file)
    tee_stderr = TeeLogger(log_file)
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    print(f"[日志系统] 日志文件: {log_file.resolve()}")
    print(f"[日志系统] 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return tee_stdout, tee_stderr


def write_experiment_log(expname, strs):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / f"{expname}.txt").open('a', encoding='utf-8') as f:
        f.write(strs)



def run_3d_ads(args):
    

    if args.dataset == 'mvtec':
        classes = mvtec3d_classes()
    if args.dataset == 'real':
        classes = real3d_classes()
    if args.dataset == 'shapenet':
        classes = shapenet3d_classes()
    if args.dataset == 'eyecandies':
        classes = eyecandies_classes() 
    if args.dataset == 'mulsen':
        classes = mulsen_classes() 
    if args.dataset == 'minishift':
        classes = minishiftAD_classes() 
    method_name = 'NCRP' if getattr(args, 'agg_method', 'mean') == 'ncrp' else 'Simple3D'
    if args.use_MSND:
        method_name += f' ({getattr(args, "msnd_impl", "official")} MSND)'
    METHOD_NAMES = [method_name]

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    for category in classes:
        patchcore = PatchCore(args=args)
        # patchcore.train(category)
        patchcore.fit(category)

    # for cls in args.category:
        cls = category
        print(f"\nRunning on class {cls}\n")
        write_experiment_log(args.expname,f"\nRunning on class {cls}\n")
        image_rocaucs, pixel_rocaucs, au_pros = patchcore.evaluate(cls)
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map({method_name: image_rocaucs['Simple3D']})
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map({method_name: pixel_rocaucs['Simple3D']})
        if args.dataset == 'real':
            au_pros_df[cls.title()] = 'Not calculated'
        else:
            au_pros_df[cls.title()] = au_pros_df['Method'].map({method_name: au_pros['Simple3D']})

        # 立即输出当前类别的结果
        print(f"\n{'='*80}")
        print(f"Finished class {cls}")
        print(f"{'='*80}")
        print(f"{method_name} O-AUROC: {image_rocaucs['Simple3D']:.3f}")
        print(f"{method_name} P-AUROC: {pixel_rocaucs['Simple3D']:.3f}")
        print(f"{'='*80}\n")

        result_str = (f"\n{'='*80}\n"
                     f"Finished class {cls}\n"
                     f"{'='*80}\n"
                     f"{method_name} O-AUROC: {image_rocaucs['Simple3D']:.3f}\n"
                     f"{method_name} P-AUROC: {pixel_rocaucs['Simple3D']:.3f}\n"
                     f"{'='*80}\n\n")
        write_experiment_log(args.expname, result_str)
        # write_experiment_log(args.expname,"################################################################################\n\n")
        

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    if args.dataset == 'real':
        au_pros_df['Mean'] = 'Not calculated'
    else:
        au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),3)

    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,image_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')

    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,pixel_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))
    write_experiment_log(args.expname,au_pros_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')


if __name__ == '__main__':
    # 启动日志系统
    tee_stdout, tee_stderr = setup_logging()

    try:
        parser = argparse.ArgumentParser(description='Process some integers.')
        parser.add_argument('--expname', type=str, default='None', help='expname')
        parser.add_argument('--device', type=str, default='cuda:0', help='expname')
        parser.add_argument('--dataset', type=str, default='shapenet', help='dataset name')
        parser.add_argument('--max_nn', type=int, default=None, help='max_nn')
        parser.add_argument('--num_group', type=int, default=None, help='num_group')
        parser.add_argument('--group_size', type=int, default=128, help='group_size')
        parser.add_argument('--agg_method', type=str, choices=['mean', 'ncrp'], default='mean',
                            help='mean: original Simple3D; ncrp: residual pooling (requires --use_LFSA True)')
        parser.add_argument('--msnd_impl', choices=['official', 'fixed'], default='official')
        parser.add_argument('--use_MSND', type=parse_bool, default=None)
        parser.add_argument('--use_LFSA', type=parse_bool, default=None)
        parser.add_argument('--vis_save', type=parse_bool, default=False)
        parser.add_argument('--num_MSND', type=int, default=2)
        parser.add_argument('--feature', type=str, default='FPFH')
        parser.add_argument('--level', type=str, default='ALL')
        parser.add_argument('--official_real3d', action='store_true',
                            help='Default to the official Real3D-AD settings and preprocessed format')
        parser.add_argument('--real3d_format', choices=['pcd', 'official'], default=None,
                            help='PCD server data or the author\'s preprocessed ASC/TXT data')
        parser.add_argument('--real3d_root', type=str, default=None,
                            help='Root of the official preprocessed Real3D-AD dataset')
        args = parser.parse_args()
        if args.official_real3d:
            args.dataset = 'real'
        if args.num_group is None:
            args.num_group = 4096 if args.official_real3d else 2048
        if args.max_nn is None:
            args.max_nn = 40 if args.official_real3d else 100
        if args.use_MSND is None:
            args.use_MSND = args.official_real3d
        if args.use_LFSA is None:
            args.use_LFSA = args.official_real3d
        if args.real3d_format is None:
            args.real3d_format = 'official' if args.official_real3d else 'pcd'
        if args.agg_method == 'ncrp' and not args.use_LFSA:
            parser.error('NCRP requires --use_LFSA True; use the same LFSA setting for Mean ablations.')
        print(args)

        run_3d_ads(args)

        print("=" * 80)
        print(f"[日志系统] 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        # 捕获并记录完整异常信息
        import traceback
        print("\n" + "=" * 80)
        print("[错误] 程序异常退出:")
        print("=" * 80)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise

    finally:
        # 恢复标准输出并关闭日志文件
        sys.stdout.flush()
        sys.stderr.flush()
        tee_stdout.close()
        tee_stderr.close()
