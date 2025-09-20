from http.server import HTTPServer, SimpleHTTPRequestHandler
import ssl
import os
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import serialization
import datetime
import ipaddress
import mimetypes  # 新增：用于更准确地识别文件类型

class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()
    
    def guess_type(self, path):
        # 重写此方法以更准确地设置内容类型
        base, ext = os.path.splitext(path)
        if ext in ['.js', '.html', '.css', '.txt', '.json', '.xml', '.svg']:
            # 对于文本文件，添加UTF-8编码
            ctype = mimetypes.guess_type(path)[0] or 'text/plain'
            if ctype.startswith('text/'):
                ctype += '; charset=utf-8'
            return ctype
        else:
            # 对于其他文件，使用默认猜测
            return super().guess_type(path)
    
    def send_head(self):
        # 重写send_head方法以确保正确设置内容类型
        path = self.translate_path(self.path)
        f = None
        if os.path.isdir(path):
            if not self.path.endswith('/'):
                # 重定向浏览器 - 基本上与原始代码相同
                self.send_response(301)
                self.send_header("Location", self.path + "/")
                self.end_headers()
                return None
            for index in "index.html", "index.htm":
                index = os.path.join(path, index)
                if os.path.exists(index):
                    path = index
                    break
            else:
                return self.list_directory(path)
        
        ctype = self.guess_type(path)
        try:
            f = open(path, 'rb')
        except OSError:
            self.send_error(404, "File not found")
            return None
        
        try:
            self.send_response(200)
            self.send_header("Content-type", ctype)
            fs = os.fstat(f.fileno())
            self.send_header("Content-Length", str(fs[6]))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.end_headers()
            return f
        except:
            f.close()
            raise

def generate_ssl_certificates():
    """自动生成自签名SSL证书和私钥文件（修复时间戳和IP地址问题）"""
    if os.path.exists("cert.pem") and os.path.exists("key.pem"):
        return True

    try:
        # 生成私钥
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # 修复：使用时区感知的UTC时间（替代deprecated的utcnow()）
        valid_from = datetime.datetime.now(datetime.timezone.utc)
        valid_to = valid_from + datetime.timedelta(days=365)

        # 创建证书主题和颁发者
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "CN"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "State"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "City"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Organization"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])

        # 修复：将IP地址字符串转换为ipaddress对象
        ip_addr = ipaddress.ip_address("127.0.0.1")  # 正确处理IP地址格式

        # 构建证书
        certificate = x509.CertificateBuilder() \
            .subject_name(subject) \
            .issuer_name(issuer) \
            .public_key(private_key.public_key()) \
            .serial_number(x509.random_serial_number()) \
            .not_valid_before(valid_from) \
            .not_valid_after(valid_to) \
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(ip_addr)  # 使用正确的IP地址对象
                ]),
                critical=False,
            ) \
            .sign(private_key, hashes.SHA256(), default_backend())

        # 保存私钥
        with open("key.pem", "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        # 保存证书
        with open("cert.pem", "wb") as f:
            f.write(certificate.public_bytes(serialization.Encoding.PEM))

        print("成功生成 SSL 证书和私钥文件 (cert.pem 和 key.pem)")
        return True

    except Exception as e:
        print(f"生成 SSL 证书失败: {str(e)}")
        return False

if __name__ == '__main__':
    if not generate_ssl_certificates():
        exit(1)

    host = '192.168.153.161'
    port = 443  # 建议测试时改用8443避免权限问题

    httpd = HTTPServer((host, port), CORSRequestHandler)

    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    try:
        context.load_cert_chain(certfile="cert.pem", keyfile="key.pem")
    except Exception as e:
        print(f"加载证书失败: {str(e)}")
        exit(1)

    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)

    print(f"HTTPS 服务器运行在 https://{host}:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("服务器已停止")
        httpd.server_close()