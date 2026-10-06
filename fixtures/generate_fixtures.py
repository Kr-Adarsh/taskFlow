"""
Generate synthetic documents for Operon.
Creates searchable PDF and text fixtures in fixtures/documents/.
"""

from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "documents"

def create_searchable_pdf(filepath: Path, lines: list[str]) -> None:
    """Creates a standard PDF-1.4 file with searchable Helvetica text stream."""
    stream_lines = ["BT", "/F1 12 Tf", "50 720 Td", "18 TL"]
    for line in lines:
        safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream_lines.append(f"({safe_line}) Tj")
        stream_lines.append("T*")
    stream_lines.append("ET")
    
    stream_body = "\n".join(stream_lines).encode("latin1")
    stream_len = len(stream_body)
    
    obj1 = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    obj2 = b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    obj3 = b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
    obj4 = b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
    obj5_hdr = f"5 0 obj\n<< /Length {stream_len} >>\nstream\n".encode("latin1")
    obj5_ftr = b"\nendstream\nendobj\n"
    
    header = b"%PDF-1.4\n"
    offset1 = len(header)
    offset2 = offset1 + len(obj1)
    offset3 = offset2 + len(obj2)
    offset4 = offset3 + len(obj3)
    offset5 = offset4 + len(obj4)
    xref_offset = offset5 + len(obj5_hdr) + stream_len + len(obj5_ftr)
    
    xref = (
        f"xref\n0 6\n"
        f"0000000000 65535 f \n"
        f"{offset1:010d} 00000 n \n"
        f"{offset2:010d} 00000 n \n"
        f"{offset3:010d} 00000 n \n"
        f"{offset4:010d} 00000 n \n"
        f"{offset5:010d} 00000 n \n"
        f"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        f"startxref\n{xref_offset}\n%%EOF\n"
    ).encode("latin1")
    
    pdf_bytes = header + obj1 + obj2 + obj3 + obj4 + obj5_hdr + stream_body + obj5_ftr + xref
    filepath.write_bytes(pdf_bytes)

def generate_all():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Acme Invoice 1021 (Older invoice)
    acme_1021_lines = [
        "ACME CORPORATION - OFFICIAL INVOICE",
        "Vendor: Acme Corp",
        "Invoice Number: INV-1021",
        "Invoice Date: 2026-08-10",
        "Due Date: 2026-09-10",
        "Billing Currency: INR",
        "Subtotal: INR 42,000",
        "Tax (0%): INR 0",
        "Total Amount: INR 42,000",
        "Payment Terms: Net 30",
        "Status: Settled"
    ]
    create_searchable_pdf(FIXTURES_DIR / "acme_invoice_1021.pdf", acme_1021_lines)

    # 2. Acme Invoice 1044 (LATEST invoice - INR 84,500 due 2026-10-15)
    acme_1044_lines = [
        "ACME CORPORATION - COMMERCIAL INVOICE",
        "Vendor: Acme Corp",
        "Invoice Number: INV-1044",
        "Invoice Date: 2026-09-15",
        "Due Date: 2026-10-15",
        "Billing Currency: INR",
        "Description: Managed Cloud Database Clusters - Premium Enterprise",
        "Amount Due: INR 84,500",
        "Payment Terms: Net 30",
        "Bank Details: HDFC Bank - ACME OPERATING AC 992019481",
        "Notes: Latest Q3 recurring invoice."
    ]
    create_searchable_pdf(FIXTURES_DIR / "acme_invoice_1044.pdf", acme_1044_lines)

    # 3. Acme Invoice 1005 (Text file format)
    (FIXTURES_DIR / "acme_invoice_1005.txt").write_text(
        "VENDOR: Acme Corp\n"
        "INVOICE NUMBER: INV-1005\n"
        "INVOICE DATE: 2026-07-01\n"
        "DUE DATE: 2026-07-31\n"
        "CURRENCY: INR\n"
        "AMOUNT: 30,000\n"
        "STATUS: Archived\n",
        encoding="utf-8"
    )

    # 4. Globex Invoice 301 (Different vendor)
    globex_301_lines = [
        "GLOBEX INDUSTRIES - INVOICE",
        "Vendor: Globex Inc",
        "Invoice Number: GLX-301",
        "Invoice Date: 2026-09-20",
        "Due Date: 2026-10-20",
        "Billing Currency: INR",
        "Amount: INR 112,000",
        "Status: Unpaid"
    ]
    create_searchable_pdf(FIXTURES_DIR / "globex_invoice_301.pdf", globex_301_lines)

    # 5. Complaint 4821 (Enterprise customer Acme Corp)
    (FIXTURES_DIR / "complaint_4821.txt").write_text(
        "CUSTOMER COMPLAINT REPORT #4821\n"
        "Date Logged: 2026-10-02 09:14 UTC\n"
        "Customer Name: Acme Corp\n"
        "Contact Person: Sarah Jenkins (VP Operations, sjenkins@acme.com)\n"
        "Account Reference: ACME-ENT-001\n"
        "Severity Reported: Critical Outage\n"
        "Details:\n"
        "We are experiencing severe database synchronization outages across our EU servers\n"
        "affecting hundreds of our enterprise users. Our internal analytics have completely stalled.\n"
        "Need urgent resolution and senior engineering support immediately.\n",
        encoding="utf-8"
    )

    # 6. Complaint 4822 (Starter customer Beta Retail)
    (FIXTURES_DIR / "complaint_4822.txt").write_text(
        "CUSTOMER COMPLAINT REPORT #4822\n"
        "Date Logged: 2026-10-03 14:22 UTC\n"
        "Customer Name: Beta Retail\n"
        "Contact Person: Dave Miller (dave@betaretail.example)\n"
        "Account Reference: BETA-STR-042\n"
        "Severity Reported: Minor UI Bug\n"
        "Details:\n"
        "Small UI alignment glitch on our mobile checkout button. Not urgent, just flagging\n"
        "for the next regular maintenance sprint.\n",
        encoding="utf-8"
    )

    print("All fixtures generated successfully.")

if __name__ == "__main__":
    generate_all()
