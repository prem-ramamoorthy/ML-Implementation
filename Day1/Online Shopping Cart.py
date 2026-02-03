from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
import json
import os
import random
from typing import Dict, List, Optional

@dataclass
class Product:
    id: str
    name: str
    price: float
    stock: int

    def is_available(self, qty: int) -> bool:
        return qty > 0 and self.stock >= qty

    def reduce_stock(self, qty: int) -> None:
        if qty <= 0:
            raise ValueError("Quantity must be > 0")
        if self.stock < qty:
            raise ValueError("Not enough stock")
        self.stock -= qty

    def increase_stock(self, qty: int) -> None:
        if qty <= 0:
            raise ValueError("Quantity must be > 0")
        self.stock += qty

@dataclass
class CartItem:
    product: Product
    quantity: int

    def subtotal(self) -> float:
        return self.product.price * self.quantity


class Cart:
    def __init__(self) -> None:
        self._items: Dict[str, CartItem] = {}

    def add_product(self, product: Product, qty: int) -> None:
        if qty <= 0:
            raise ValueError("Quantity must be > 0")
        current = self._items.get(product.id)
        new_qty = qty if current is None else current.quantity + qty

        if not product.is_available(new_qty):
            raise ValueError(f"Not enough stock for '{product.name}'. Available: {product.stock}")

        self._items[product.id] = CartItem(product=product, quantity=new_qty)

    def remove_product(self, product_id: str) -> None:
        if product_id not in self._items:
            raise KeyError("Product not in cart")
        del self._items[product_id]

    def update_quantity(self, product_id: str, qty: int) -> None:
        if qty < 0:
            raise ValueError("Quantity cannot be negative")
        if product_id not in self._items:
            raise KeyError("Product not in cart")

        if qty == 0:
            del self._items[product_id]
            return

        item = self._items[product_id]
        if not item.product.is_available(qty):
            raise ValueError(f"Not enough stock for '{item.product.name}'. Available: {item.product.stock}")

        item.quantity = qty

    def clear(self) -> None:
        self._items.clear()

    def total(self) -> float:
        return sum(item.subtotal() for item in self._items.values())

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def items_list(self) -> List[CartItem]:
        return list(self._items.values())

    def view(self) -> str:
        if self.is_empty():
            return "Cart is empty."

        lines: List[str] = []
        lines.append("\n--- CART ---")
        lines.append(f"{'ID':<6} {'Product':<18} {'Qty':<5} {'Price':<10} {'Subtotal':<10}")
        lines.append("-" * 55)
        for it in self._items.values():
            lines.append(
                f"{it.product.id:<6} {it.product.name:<18} {it.quantity:<5} "
                f"₹{it.product.price:<9.2f} ₹{it.subtotal():<9.2f}"
            )
        lines.append("-" * 55)
        lines.append(f"Total: ₹{self.total():.2f}\n")
        return "\n".join(lines)


@dataclass
class Customer:
    name: str
    email: str
    address: str

class Discount(ABC):
    @abstractmethod
    def apply(self, total: float) -> float:
        pass

    @abstractmethod
    def label(self) -> str:
        pass


class NoDiscount(Discount):
    def apply(self, total: float) -> float:
        return total

    def label(self) -> str:
        return "No Discount"


class FlatDiscount(Discount):
    def __init__(self, amount: float) -> None:
        self.amount = max(0.0, float(amount))

    def apply(self, total: float) -> float:
        return max(0.0, total - self.amount)

    def label(self) -> str:
        return f"Flat -₹{self.amount:.2f}"


class PercentageDiscount(Discount):
    def __init__(self, percent: float) -> None:
        self.percent = max(0.0, min(100.0, float(percent)))

    def apply(self, total: float) -> float:
        return max(0.0, total * (1.0 - self.percent / 100.0))

    def label(self) -> str:
        return f"{self.percent:.0f}% Off"


class CouponDiscount(Discount):
    def __init__(self, code: str, percent: float, min_amount: float) -> None:
        self.code = code.strip().upper()
        self.percent = max(0.0, min(100.0, float(percent)))
        self.min_amount = max(0.0, float(min_amount))

    def apply(self, total: float) -> float:
        if total < self.min_amount:
            return total
        return max(0.0, total * (1.0 - self.percent / 100.0))

    def label(self) -> str:
        return f"Coupon {self.code} ({self.percent:.0f}% Off, min ₹{self.min_amount:.2f})"

class PaymentMethod(ABC):
    @abstractmethod
    def pay(self, amount: float) -> bool:
        """Return True if payment succeeds."""
        pass

    @abstractmethod
    def label(self) -> str:
        pass


class UPIPayment(PaymentMethod):
    def __init__(self, upi_id: str) -> None:
        self.upi_id = upi_id.strip()

    def pay(self, amount: float) -> bool:
        return amount >= 0

    def label(self) -> str:
        return f"UPI ({self.upi_id})"


class CardPayment(PaymentMethod):
    def __init__(self, last4: str) -> None:
        last4 = last4.strip()
        if len(last4) != 4 or not last4.isdigit():
            raise ValueError("Card last4 must be 4 digits")
        self.last4 = last4

    def pay(self, amount: float) -> bool:
        return amount >= 0

    def label(self) -> str:
        return f"Card (**** {self.last4})"


class CashOnDelivery(PaymentMethod):
    def pay(self, amount: float) -> bool:
        return amount >= 0

    def label(self) -> str:
        return "Cash on Delivery"

@dataclass
class Order:
    order_id: str
    customer: Customer
    items: List[CartItem]
    subtotal: float
    discount_label: str
    total: float
    payment_label: str
    paid: bool
    created_at: str

    def invoice_text(self) -> str:
        lines: List[str] = []
        lines.append("\n================= INVOICE =================")
        lines.append(f"Order ID   : {self.order_id}")
        lines.append(f"Date/Time  : {self.created_at}")
        lines.append(f"Customer   : {self.customer.name} ({self.customer.email})")
        lines.append(f"Address    : {self.customer.address}")
        lines.append("-------------------------------------------")
        lines.append(f"{'Product':<22} {'Qty':<5} {'Price':<10} {'Subtotal':<10}")
        lines.append("-------------------------------------------")
        for it in self.items:
            lines.append(
                f"{it.product.name:<22} {it.quantity:<5} ₹{it.product.price:<9.2f} ₹{it.subtotal():<9.2f}"
            )
        lines.append("-------------------------------------------")
        lines.append(f"Subtotal   : ₹{self.subtotal:.2f}")
        lines.append(f"Discount   : {self.discount_label}")
        lines.append(f"Total      : ₹{self.total:.2f}")
        lines.append(f"Payment    : {self.payment_label}")
        lines.append(f"Status     : {'PAID' if self.paid else 'FAILED'}")
        lines.append("===========================================\n")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "created_at": self.created_at,
            "customer": {
                "name": self.customer.name,
                "email": self.customer.email,
                "address": self.customer.address,
            },
            "items": [
                {
                    "product_id": it.product.id,
                    "name": it.product.name,
                    "price": it.product.price,
                    "quantity": it.quantity,
                    "subtotal": it.subtotal(),
                }
                for it in self.items
            ],
            "subtotal": self.subtotal,
            "discount_label": self.discount_label,
            "total": self.total,
            "payment_label": self.payment_label,
            "paid": self.paid,
        }


class Store:
    def __init__(self, products: List[Product], orders_path: str = "orders.json") -> None:
        self.products: Dict[str, Product] = {p.id: p for p in products}
        self.orders_path = orders_path

    def list_products_text(self) -> str:
        lines: List[str] = []
        lines.append("\n--- PRODUCTS ---")
        lines.append(f"{'ID':<6} {'Name':<18} {'Price':<10} {'Stock':<6}")
        lines.append("-" * 45)
        for p in self.products.values():
            lines.append(f"{p.id:<6} {p.name:<18} ₹{p.price:<9.2f} {p.stock:<6}")
        lines.append("-" * 45)
        return "\n".join(lines)

    def get_product(self, product_id: str) -> Product:
        if product_id not in self.products:
            raise KeyError("Invalid product ID")
        return self.products[product_id]

    def checkout(
        self,
        cart: Cart,
        customer: Customer,
        discount: Discount,
        payment: PaymentMethod,
    ) -> Order:
        if cart.is_empty():
            raise ValueError("Cart is empty")

        for it in cart.items_list():
            if not it.product.is_available(it.quantity):
                raise ValueError(f"Stock changed. '{it.product.name}' available: {it.product.stock}")

        subtotal = cart.total()
        total_after_discount = discount.apply(subtotal)

        for it in cart.items_list():
            it.product.reduce_stock(it.quantity)

        paid = payment.pay(total_after_discount)

        order = Order(
            order_id=self._generate_order_id(),
            customer=customer,
            items=cart.items_list(),
            subtotal=subtotal,
            discount_label=discount.label(),
            total=total_after_discount,
            payment_label=payment.label(),
            paid=paid,
            created_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        self._save_order(order)
        cart.clear()
        return order

    def _generate_order_id(self) -> str:
        return f"ORD{random.randint(10000, 99999)}"

    def _save_order(self, order: Order) -> None:
        data: List[dict] = []
        if os.path.exists(self.orders_path):
            try:
                with open(self.orders_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except Exception:
                data = []
        data.append(order.to_dict())
        with open(self.orders_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def pick_discount() -> Discount:
    print("\nChoose discount:")
    print("1) No Discount")
    print("2) Flat Discount (e.g., ₹100 off)")
    print("3) Percentage Discount (e.g., 10% off)")
    print("4) Coupon (SAVE10 -> 10% off, min ₹500)")
    choice = input("Enter choice: ").strip()

    if choice == "2":
        amt = float(input("Enter flat amount: ").strip())
        return FlatDiscount(amt)
    if choice == "3":
        pct = float(input("Enter percent: ").strip())
        return PercentageDiscount(pct)
    if choice == "4":
        code = input("Enter coupon code: ").strip().upper()
        if code == "SAVE10":
            return CouponDiscount(code="SAVE10", percent=10, min_amount=500)
        print("Invalid coupon. Applying no discount.")
        return NoDiscount()
    return NoDiscount()


def pick_payment() -> PaymentMethod:
    print("\nChoose payment method:")
    print("1) UPI")
    print("2) Card")
    print("3) Cash on Delivery")
    choice = input("Enter choice: ").strip()

    if choice == "1":
        upi = input("Enter UPI ID (e.g., prem@upi): ").strip()
        return UPIPayment(upi)
    if choice == "2":
        last4 = input("Enter last 4 digits of card: ").strip()
        return CardPayment(last4)
    return CashOnDelivery()


def main() -> None:
    products = [
        Product("P101", "Mouse", 399.0, 20),
        Product("P102", "Keyboard", 1199.0, 10),
        Product("P103", "USB-C Cable", 199.0, 30),
        Product("P104", "Headphones", 1499.0, 8),
        Product("P105", "Laptop Stand", 899.0, 12),
    ]

    store = Store(products=products, orders_path="orders.json")
    cart = Cart()

    print("=== ONLINE SHOPPING CART (OOP) ===")
    print("Author: Prem Kumar R | Reg.No: 23MIP10019")

    while True:
        print("\nMenu:")
        print("1) View Products")
        print("2) Add to Cart")
        print("3) View Cart")
        print("4) Update Quantity")
        print("5) Remove from Cart")
        print("6) Checkout")
        print("7) Exit")

        option = input("Choose: ").strip()

        try:
            if option == "1":
                print(store.list_products_text())

            elif option == "2":
                pid = input("Enter Product ID: ").strip()
                qty = int(input("Enter Quantity: ").strip())
                product = store.get_product(pid)
                cart.add_product(product, qty)
                print("Added to cart.")

            elif option == "3":
                print(cart.view())

            elif option == "4":
                pid = input("Enter Product ID: ").strip()
                qty = int(input("Enter New Quantity (0 to remove): ").strip())
                cart.update_quantity(pid, qty)
                print("Cart updated.")

            elif option == "5":
                pid = input("Enter Product ID to remove: ").strip()
                cart.remove_product(pid)
                print("Removed from cart.")

            elif option == "6":
                if cart.is_empty():
                    print("Cart is empty. Add items first.")
                    continue

                name = input("Customer Name: ").strip()
                email = input("Email: ").strip()
                address = input("Address: ").strip()
                customer = Customer(name=name, email=email, address=address)

                discount = pick_discount()
                payment = pick_payment()

                order = store.checkout(cart, customer, discount, payment)
                print(order.invoice_text())
                print("Order saved to orders.json")

            elif option == "7":
                print("Bye 👋")
                break

            else:
                print("Invalid option.")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()