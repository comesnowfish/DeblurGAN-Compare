import dominate
from dominate.tags import *
import os
import sys


class HTML:
    def __init__(self, web_dir, title, refresh=0):
        """初始化HTML报告
        Args:
            web_dir: 保存目录路径 (e.g. 'results/experiment1')
            title: 网页标题
            refresh: 自动刷新时间（秒）
        """
        self.title = title
        self.web_dir = os.path.abspath(web_dir)  # 转为绝对路径
        self.img_dir = os.path.join(self.web_dir, 'images')

        # 调试输出
        print(f"\n[HTML Builder] 初始化目录: {self.web_dir}")

        # 创建目录
        os.makedirs(self.web_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

        # 初始化DOM文档
        self.doc = dominate.document(title=title)
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="refresh", content=str(refresh))

        # 添加基础样式
        with self.doc.head:
            style("""
                table { width: 100%; border-collapse: collapse; }
                td { padding: 8px; text-align: center; border: 1px solid #ddd; }
                img { max-width: 100%; height: auto; }
            """)

    def add_header(self, text):
        """添加标题"""
        with self.doc:
            h1(text)
        print(f"[HTML] 添加标题: '{text}'")

    def add_images(self, ims, txts, links=None, width=400):
        """批量添加图片
        Args:
            ims: 图片文件名列表 (e.g. ['result1.png'])
            txts: 描述文字列表
            links: 点击链接列表（默认同图片名）
            width: 图片显示宽度
        """
        if links is None:
            links = ims

        # 调试输出
        print(f"[HTML] 添加 {len(ims)} 张图片到表格")

        # 创建表格
        if not hasattr(self, 't'):
            self.add_table()

        # 添加图片行
        with self.t:
            with tr():
                for im, txt, link in zip(ims, txts, links):
                    img_path = os.path.join('images', im)
                    with td(style="word-wrap: break-word;"):
                        with p():
                            # 确保图片文件存在
                            if not os.path.exists(os.path.join(self.web_dir, img_path)):
                                print(f"! 警告: 图片不存在 {img_path}", file=sys.stderr)

                            a(img(src=img_path, style=f"width:{width}px"),
                              href=os.path.join('images', link))
                            br()
                            p(txt)

    def add_table(self, border=1):
        """初始化表格"""
        self.t = table(border=border, style="table-layout: fixed;")
        self.doc.add(self.t)
        print("[HTML] 创建新表格")

    def save(self):
        """保存HTML文件"""
        html_file = os.path.join(self.web_dir, 'index.html')

        try:
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(self.doc.render())
            print(f"[HTML] 成功保存报告到 {html_file}")
            return True
        except Exception as e:
            print(f"! 保存HTML失败: {str(e)}", file=sys.stderr)
            return False


# 测试用例
if __name__ == '__main__':
    # 1. 初始化
    html = HTML('web_test', '测试报告')

    # 2. 添加内容
    html.add_header('去模糊结果对比')

    # 3. 模拟添加图片
    test_images = [
        ('blurred.png', '模糊输入'),
        ('deblurred.png', '去模糊结果'),
        ('ground_truth.png', '真实图像')
    ]

    # 创建测试图片（空文件）
    for img, _ in test_images:
        open(os.path.join(html.img_dir, img), 'w').close()

    # 添加到HTML
    html.add_images(
        ims=[i[0] for i in test_images],
        txts=[i[1] for i in test_images]
    )

    # 4. 保存
    html.save()