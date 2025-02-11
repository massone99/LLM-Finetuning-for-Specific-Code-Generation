import akka.actor.{Actor, ActorSystem, Props}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

// Message to initiate a payment
case class PaymentRequest(orderId: String, amount: Double)
// Confirmation and Failure messages
case class PaymentConfirmation(orderId: String)
case class PaymentFailure(orderId: String, reason: String)

// Actor that simulates payment processing
class PaymentProcessorActor extends Actor {
  def receive: Receive = {
    case PaymentRequest(orderId, amount) =>
      // Log receipt of the payment request
      println(s"PaymentProcessorActor: Processing payment for order $orderId, amount $$ $amount")
      // Simulate asynchronous processing with a short delay
      context.system.scheduler.scheduleOnce(1.second) {
        if (amount < 1000.0) {
          println(s"PaymentProcessorActor: Payment for order $orderId is confirmed.")
          sender() ! PaymentConfirmation(orderId)
        } else {
          println(s"PaymentProcessorActor: Payment for order $orderId failed (insufficient funds).")
          sender() ! PaymentFailure(orderId, "Insufficient funds")
        }
      }
    case _ =>
      // Log unknown messages.
      println("PaymentProcessorActor: Unknown message.")
  }
}

object PaymentProcessorApp extends App {
  // Create an ActorSystem named "PaymentSystem"
  val system = ActorSystem("PaymentSystem")
  // Create an instance of PaymentProcessorActor
  val processor = system.actorOf(Props[PaymentProcessorActor](), "paymentProcessor")

  // Send payment requests
  processor ! PaymentRequest("Order001", 500.0)
  processor ! PaymentRequest("Order002", 1500.0)

  // Wait briefly for processing, then terminate
  Thread.sleep(2000)
  system.terminate()
}